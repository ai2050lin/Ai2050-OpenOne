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
import phase801_target_neutral_suppressor_causal_test as p801  # noqa: E402
import phase802_new_blocker_stabilization_dose_response as p802  # noqa: E402
import phase803_semantic_new_blocker_source_localization as p803  # noqa: E402
import phase804_true_semantic_suppressor_projection_search as p804  # noqa: E402
import phase805_residual_closure_blocker_audit as p805  # noqa: E402
import phase806_format_echo_identity_residual_suppressor_search as p806  # noqa: E402
import phase807_readout_geometry_new_blocker_emergence_audit as p807  # noqa: E402
from model_utils import release_model  # noqa: E402
from phase722_functional_head_atlas_causal_ablation import write_json, write_jsonl  # noqa: E402
from phase735_source_restricted_writer_validation import MODELS  # noqa: E402
from phase771_matched_causal_intervention_reliability_test import case_map_for  # noqa: E402
from phase773_instruction_source_disentanglement import fmt  # noqa: E402
from phase780_surface_form_component_localization import lm_head_weight  # noqa: E402
from phase795_multi_component_causal_fiber_closure import selected_route_components  # noqa: E402


RESULT_ROOT = Path("tests/result/phase808_readout_closer_source_localization")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def safe_float(value: Any) -> float | None:
    return p805.safe_float(value)


def safe_mean(values: list[Any]) -> float | None:
    return p805.safe_mean(values)


def safe_rate(values: list[Any]) -> float | None:
    return p805.safe_rate(values)


def parse_float_grid(text: str, fallback: list[float]) -> list[float]:
    return p805.parse_float_grid(text, fallback)


def finite(value: Any, default: float = 0.0) -> float:
    val = safe_float(value)
    return default if val is None else val


def component_label(key: tuple[str, int]) -> str:
    return f"{key[0]}:L{int(key[1])}"


def transition_prefixed(
    prefix: str,
    args: argparse.Namespace,
    base_logits: torch.Tensor,
    after_logits: torch.Tensor,
    target_id: int,
    contrast_id: int,
    prompt_ids: list[int],
    prompt_text: str,
    candidate_ids: set[int],
    case_values: set[str],
) -> dict[str, Any]:
    metrics = p807.transition_metrics(
        args,
        base_logits,
        after_logits,
        target_id,
        contrast_id,
        prompt_ids,
        prompt_text,
        candidate_ids,
        case_values,
    )
    return {f"{prefix}_{k}": v for k, v in metrics.items()}


def snapshot_prefixed(
    prefix: str,
    args: argparse.Namespace,
    logits: torch.Tensor,
    target_id: int,
    contrast_id: int,
    prompt_ids: list[int],
    prompt_text: str,
    candidate_ids: set[int],
    case_values: set[str],
    target_answer: str,
) -> dict[str, Any]:
    metrics = p805.residual_snapshot(
        args,
        logits,
        target_id,
        contrast_id,
        prompt_ids,
        prompt_text,
        candidate_ids,
        case_values,
        target_answer,
    )
    return {f"{prefix}_{k}": v for k, v in metrics.items()}


def label_component(row: dict[str, Any], args: argparse.Namespace) -> str:
    single_net = finite(row.get("single_transition_net_count_delta"))
    single_emerged = finite(row.get("single_transition_emerged_count"))
    single_resolved = finite(row.get("single_transition_resolved_count"))
    single_emergence_rate = finite(row.get("single_transition_emergence_rate_vs_base"))
    single_bias = finite(row.get("single_required_bias_delta_vs_semantic_base"))
    loo_net_loss = finite(row.get("loo_net_loss_vs_full"))
    loo_bias_loss = finite(row.get("loo_bias_loss_vs_full"))
    if bool(row.get("single_token_closure_gain")):
        return "single_component_token_closure"
    if (
        single_net < 0
        and single_resolved > single_emerged
        and single_emergence_rate <= float(args.max_emergence_rate_for_source)
        and single_bias <= 0
        and loo_net_loss >= float(args.min_loo_net_loss)
    ):
        return "source_closer_candidate_no_closure"
    if loo_net_loss >= float(args.min_loo_net_loss) and loo_bias_loss >= float(args.min_loo_bias_loss):
        return "distributed_closer_contributor"
    if single_emerged > single_resolved and single_net > 0:
        return "new_blocker_source_or_field_deformer"
    if single_net < 0 and single_resolved > single_emerged:
        return "weak_local_reducer"
    return "neutral_or_weak"


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

    alpha0_projected, _alpha0_metrics = p802.projected_component_state_alpha(
        recipient_state,
        donor_state,
        route_components,
        target_direction,
        0.0,
    )
    alpha0_logits, alpha0_error = p801.run_logits_with_projected_route(model, device, recipient_ids, alpha0_projected, recipient_answer_pos)
    if alpha0_error or not alpha0_logits.numel():
        return [
            {
                "row_kind": "phase808_error",
                "model": args.model,
                "case_id": case["case_id"],
                "route_id": route["route_id"],
                "error": alpha0_error or "empty_alpha0_logits",
            }
        ]
    alpha0_semantic = p803.semantic_new_blocker_ids(
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
    semantic_direction = p804.semantic_direction_from_blockers(unembed, target_id, alpha0_semantic, args.semantic_direction_mode)
    rows: list[dict[str, Any]] = []
    for target_alpha in parse_float_grid(args.target_alpha_grid, [0.75]):
        semantic_projected, semantic_metrics = p804.projected_state_target_semantic(
            recipient_state,
            donor_state,
            route_components,
            target_direction,
            semantic_direction,
            target_alpha,
            float(args.semantic_beta),
        )
        semantic_logits, semantic_error = p801.run_logits_with_projected_route(
            model,
            device,
            recipient_ids,
            semantic_projected,
            recipient_answer_pos,
        )
        if semantic_error or not semantic_logits.numel():
            rows.append(
                {
                    "row_kind": "phase808_error",
                    "model": args.model,
                    "case_id": case["case_id"],
                    "route_id": route["route_id"],
                    "target_direction_alpha": float(target_alpha),
                    "semantic_suppression_beta": float(args.semantic_beta),
                    "error": semantic_error or "empty_semantic_base_logits",
                }
            )
            continue
        direction_tokens = p806.collect_residual_direction_tokens(
            args,
            semantic_logits,
            target_id,
            contrast_id,
            recipient_ids,
            recipient_prompt,
            recipient_candidate_ids,
            case_vals,
            str(case.get("answer", "")),
        )
        format_direction = p806.direction_from_ids(unembed, target_id, direction_tokens["format_echo_ids"])
        identity_direction = p806.direction_from_ids(unembed, target_id, direction_tokens["identity_variant_ids"])
        full_projected, full_projection_metrics = p806.projected_state_multi(
            recipient_state,
            donor_state,
            route_components,
            target_direction,
            semantic_direction,
            format_direction,
            identity_direction,
            target_alpha,
            float(args.semantic_beta),
            float(args.format_beta),
            float(args.identity_beta),
        )
        full_logits, full_error = p801.run_logits_with_projected_route(model, device, recipient_ids, full_projected, recipient_answer_pos)
        if full_error or not full_logits.numel():
            rows.append(
                {
                    "row_kind": "phase808_error",
                    "model": args.model,
                    "case_id": case["case_id"],
                    "route_id": route["route_id"],
                    "target_direction_alpha": float(target_alpha),
                    "semantic_suppression_beta": float(args.semantic_beta),
                    "format_echo_suppression_beta": float(args.format_beta),
                    "identity_anchor_suppression_beta": float(args.identity_beta),
                    "error": full_error or "empty_full_logits",
                }
            )
            continue
        full_transition = p807.transition_metrics(
            args,
            semantic_logits,
            full_logits,
            target_id,
            contrast_id,
            recipient_ids,
            recipient_prompt,
            recipient_candidate_ids,
            case_vals,
        )
        full_snapshot = p805.residual_snapshot(
            args,
            full_logits,
            target_id,
            contrast_id,
            recipient_ids,
            recipient_prompt,
            recipient_candidate_ids,
            case_vals,
            str(case.get("answer", "")),
        )
        full_required_bias_delta = finite(full_snapshot.get("residual_required_bias_to_clear_all")) - finite(
            p805.residual_snapshot(
                args,
                semantic_logits,
                target_id,
                contrast_id,
                recipient_ids,
                recipient_prompt,
                recipient_candidate_ids,
                case_vals,
                str(case.get("answer", "")),
            ).get("residual_required_bias_to_clear_all")
        )
        route_keys = [(str(c["component_kind"]), int(c["layer"])) for c in route_components]
        for key in route_keys:
            if key not in semantic_projected or key not in full_projected:
                continue
            single_projected = dict(semantic_projected)
            single_projected[key] = full_projected[key]
            loo_projected = dict(full_projected)
            loo_projected[key] = semantic_projected[key]
            single_logits, single_error = p801.run_logits_with_projected_route(
                model,
                device,
                recipient_ids,
                single_projected,
                recipient_answer_pos,
            )
            loo_logits, loo_error = p801.run_logits_with_projected_route(
                model,
                device,
                recipient_ids,
                loo_projected,
                recipient_answer_pos,
            )
            row: dict[str, Any] = {
                "row_kind": "phase808_readout_closer_source_localization",
                "model": args.model,
                "round": args.round_name,
                "case_id": case["case_id"],
                "domain": case.get("domain"),
                "relation": case.get("relation"),
                "object": case.get("object"),
                "target_answer": case.get("answer"),
                "contrast_answer": case.get("contrast_answer"),
                "route_id": route["route_id"],
                "compare_variant": donor_variant,
                "route_component_signature": p801.route_signature(route_components),
                "component_kind": key[0],
                "layer": int(key[1]),
                "component_label": component_label(key),
                "target_direction_alpha": float(target_alpha),
                "semantic_suppression_beta": float(args.semantic_beta),
                "format_echo_suppression_beta": float(args.format_beta),
                "identity_anchor_suppression_beta": float(args.identity_beta),
                "semantic_direction_mode": args.semantic_direction_mode,
                "single_error": single_error,
                "loo_error": loo_error,
                "alpha0_semantic_new_blocker_count": len(alpha0_semantic),
                "format_echo_direction_token_ids": direction_tokens["format_echo_ids"],
                "identity_anchor_direction_token_ids": direction_tokens["identity_variant_ids"],
                "direction_class_counts": direction_tokens["direction_class_counts"],
                "component_format_delta_norm": float((full_projected[key].float() - semantic_projected[key].float()).norm().item()),
                "full_transition_net_count_delta": full_transition.get("transition_net_count_delta"),
                "full_transition_resolved_count": full_transition.get("transition_resolved_count"),
                "full_transition_emerged_count": full_transition.get("transition_emerged_count"),
                "full_transition_emergence_rate_vs_base": full_transition.get("transition_emergence_rate_vs_base"),
                "full_required_bias_delta_vs_semantic_base": full_required_bias_delta,
                "full_token_closure_gain": bool(full_snapshot.get("token_closure_gain")),
                "full_projection_mean_format_component_before": full_projection_metrics.get("mean_format_echo_component_before"),
                "full_projection_mean_format_parallel_norm": full_projection_metrics.get("mean_format_echo_parallel_norm"),
            }
            if single_logits.numel():
                row.update(
                    transition_prefixed(
                        "single",
                        args,
                        semantic_logits,
                        single_logits,
                        target_id,
                        contrast_id,
                        recipient_ids,
                        recipient_prompt,
                        recipient_candidate_ids,
                        case_vals,
                    )
                )
                single_snapshot = p805.residual_snapshot(
                    args,
                    single_logits,
                    target_id,
                    contrast_id,
                    recipient_ids,
                    recipient_prompt,
                    recipient_candidate_ids,
                    case_vals,
                    str(case.get("answer", "")),
                )
                row["single_required_bias_delta_vs_semantic_base"] = finite(
                    single_snapshot.get("residual_required_bias_to_clear_all")
                ) - finite(
                    p805.residual_snapshot(
                        args,
                        semantic_logits,
                        target_id,
                        contrast_id,
                        recipient_ids,
                        recipient_prompt,
                        recipient_candidate_ids,
                        case_vals,
                        str(case.get("answer", "")),
                    ).get("residual_required_bias_to_clear_all")
                )
                row["single_token_closure_gain"] = bool(single_snapshot.get("token_closure_gain"))
                row.update(
                    p806.tracked_token_metrics(
                        semantic_logits,
                        single_logits,
                        target_id,
                        direction_tokens["format_echo_ids"],
                        "single_format_echo",
                    )
                )
            if loo_logits.numel():
                row.update(
                    transition_prefixed(
                        "loo",
                        args,
                        semantic_logits,
                        loo_logits,
                        target_id,
                        contrast_id,
                        recipient_ids,
                        recipient_prompt,
                        recipient_candidate_ids,
                        case_vals,
                    )
                )
                loo_snapshot = p805.residual_snapshot(
                    args,
                    loo_logits,
                    target_id,
                    contrast_id,
                    recipient_ids,
                    recipient_prompt,
                    recipient_candidate_ids,
                    case_vals,
                    str(case.get("answer", "")),
                )
                loo_bias_delta = finite(loo_snapshot.get("residual_required_bias_to_clear_all")) - finite(
                    p805.residual_snapshot(
                        args,
                        semantic_logits,
                        target_id,
                        contrast_id,
                        recipient_ids,
                        recipient_prompt,
                        recipient_candidate_ids,
                        case_vals,
                        str(case.get("answer", "")),
                    ).get("residual_required_bias_to_clear_all")
                )
                row["loo_required_bias_delta_vs_semantic_base"] = loo_bias_delta
                row["loo_token_closure_gain"] = bool(loo_snapshot.get("token_closure_gain"))
                row["loo_net_loss_vs_full"] = finite(row.get("loo_transition_net_count_delta")) - finite(
                    row.get("full_transition_net_count_delta")
                )
                row["loo_resolved_loss_vs_full"] = finite(row.get("full_transition_resolved_count")) - finite(
                    row.get("loo_transition_resolved_count")
                )
                row["loo_emerged_delta_vs_full"] = finite(row.get("loo_transition_emerged_count")) - finite(
                    row.get("full_transition_emerged_count")
                )
                row["loo_bias_loss_vs_full"] = loo_bias_delta - full_required_bias_delta
            row["phase808_label"] = label_component(row, args)
            row["phase808_boundary"] = (
                "This phase localizes component-level sources of the qwen3 format-only readout closer candidate. "
                "A source candidate must reduce old blockers with low new-blocker emergence; target-logit gain alone is insufficient."
            )
            rows.append(row)
    return rows


def group_rows(rows: list[dict[str, Any]], fields: list[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("row_kind") == "phase808_readout_closer_source_localization":
            groups[tuple(row.get(f) for f in fields)].append(row)
    out: list[dict[str, Any]] = []
    for key, vals in groups.items():
        payload = {field: value for field, value in zip(fields, key)}
        payload.update(
            {
                "n": len(vals),
                "case_n": len({v.get("case_id") for v in vals}),
                "mean_component_format_delta_norm": safe_mean([v.get("component_format_delta_norm") for v in vals]),
                "mean_full_net_delta": safe_mean([v.get("full_transition_net_count_delta") for v in vals]),
                "mean_full_resolved": safe_mean([v.get("full_transition_resolved_count") for v in vals]),
                "mean_full_emerged": safe_mean([v.get("full_transition_emerged_count") for v in vals]),
                "mean_full_bias_delta": safe_mean([v.get("full_required_bias_delta_vs_semantic_base") for v in vals]),
                "mean_single_net_delta": safe_mean([v.get("single_transition_net_count_delta") for v in vals]),
                "mean_single_resolved": safe_mean([v.get("single_transition_resolved_count") for v in vals]),
                "mean_single_emerged": safe_mean([v.get("single_transition_emerged_count") for v in vals]),
                "mean_single_emergence_rate": safe_mean([v.get("single_transition_emergence_rate_vs_base") for v in vals]),
                "mean_single_bias_delta": safe_mean([v.get("single_required_bias_delta_vs_semantic_base") for v in vals]),
                "mean_single_format_suppression": safe_mean([v.get("single_format_echo_mean_suppression_vs_semantic_base") for v in vals]),
                "mean_loo_net_loss_vs_full": safe_mean([v.get("loo_net_loss_vs_full") for v in vals]),
                "mean_loo_resolved_loss_vs_full": safe_mean([v.get("loo_resolved_loss_vs_full") for v in vals]),
                "mean_loo_emerged_delta_vs_full": safe_mean([v.get("loo_emerged_delta_vs_full") for v in vals]),
                "mean_loo_bias_loss_vs_full": safe_mean([v.get("loo_bias_loss_vs_full") for v in vals]),
                "single_token_closure_gain_rate": safe_rate([v.get("single_token_closure_gain") for v in vals]),
                "loo_token_closure_gain_rate": safe_rate([v.get("loo_token_closure_gain") for v in vals]),
                "label_counts": dict(Counter(v.get("phase808_label") for v in vals)),
            }
        )
        out.append(payload)
    out.sort(
        key=lambda r: (
            r.get("single_token_closure_gain_rate") or 0.0,
            r.get("mean_loo_net_loss_vs_full") or -999999.0,
            -(r.get("mean_single_net_delta") or 999999.0),
            -(r.get("mean_single_emerged") or 999999.0),
        ),
        reverse=True,
    )
    return out


def summarize(rows: list[dict[str, Any]], args: argparse.Namespace, attn_impl: str, routes: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "phase": 808,
        "title": "Readout Closer Source Localization with New-Blocker Control",
        "model": args.model,
        "round": args.round_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "attn_implementation": attn_impl,
        "dtype": "bfloat16",
        "quantization": "off",
        "n_rows": len(rows),
        "n_cases": len({r.get("case_id") for r in rows if r.get("row_kind") == "phase808_readout_closer_source_localization"}),
        "n_routes": len(routes),
        "target_alpha_grid": parse_float_grid(args.target_alpha_grid, [0.75]),
        "semantic_beta": float(args.semantic_beta),
        "format_beta": float(args.format_beta),
        "identity_beta": float(args.identity_beta),
        "by_component": group_rows(rows, ["model", "component_kind", "layer"]),
        "by_component_route": group_rows(rows, ["model", "route_component_signature", "component_kind", "layer"])[:120],
        "by_case_component": group_rows(rows, ["model", "case_id", "component_kind", "layer"])[:120],
        "strict_boundary": (
            "This phase moves from direction-level readout closer evidence to component-level source localization. "
            "It is not neuron-level closure and does not prove a minimal sufficient circuit."
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
        f"target_alpha={args.target_alpha_grid} semantic_beta={args.semantic_beta} "
        f"format_beta={args.format_beta} identity_beta={args.identity_beta}"
    )
    if args.dry_run:
        return {"model": args.model, "round": args.round_name, "selected_cases": len(selected), "routes": routes}
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
                log(f"{args.model}: readout closer source localization {ci}/{len(selected)} cases; rows={len(rows)}")
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
    write_jsonl(out_dir / f"phase808_{args.model}_rows.jsonl", rows)
    write_json(out_dir / f"phase808_{args.model}_summary.json", summary)
    print(
        json.dumps(
            {
                "model": args.model,
                "round": args.round_name,
                "attn": attn_impl,
                "n_cases": summary["n_cases"],
                "n_rows": summary["n_rows"],
                "by_component": summary["by_component"][:12],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return summary


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        f"# Phase 808 Readout Closer Source Localization ({payload['round']})",
        "",
        f"- Status: `{payload['status']}`",
        "- Boundary: component-level source localization for the format-only closer candidate.",
        "- Success requires old-blocker reduction and low new-blocker emergence; target-logit gain alone is not enough.",
        "",
        "## By Component",
        "",
        "| model | component | rows | cases | full net | full resolved | full emerged | single net | single resolved | single emerged | single emergence rate | single bias | single fmt supp | loo net loss | loo resolved loss | loo emerged delta | loo bias loss | single closure | labels |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for model_name in MODELS:
        data = payload.get("model_summaries", {}).get(model_name)
        if not data:
            continue
        for row in data.get("by_component", [])[:80]:
            label = f"{row.get('component_kind')}:L{row.get('layer')}"
            lines.append(
                f"| {model_name} | `{label}` | {row.get('n')} | {row.get('case_n')} | "
                f"{fmt(row.get('mean_full_net_delta'))} | {fmt(row.get('mean_full_resolved'))} | {fmt(row.get('mean_full_emerged'))} | "
                f"{fmt(row.get('mean_single_net_delta'))} | {fmt(row.get('mean_single_resolved'))} | {fmt(row.get('mean_single_emerged'))} | "
                f"{fmt(row.get('mean_single_emergence_rate'))} | {fmt(row.get('mean_single_bias_delta'))} | "
                f"{fmt(row.get('mean_single_format_suppression'))} | {fmt(row.get('mean_loo_net_loss_vs_full'))} | "
                f"{fmt(row.get('mean_loo_resolved_loss_vs_full'))} | {fmt(row.get('mean_loo_emerged_delta_vs_full'))} | "
                f"{fmt(row.get('mean_loo_bias_loss_vs_full'))} | {fmt(row.get('single_token_closure_gain_rate'))} | "
                f"`{json.dumps(row.get('label_counts') or {}, ensure_ascii=False)}` |"
            )
    lines += [
        "",
        "## Top Source Candidates",
        "",
        "| model | component | single net | single emerged | single bias | loo net loss | label counts |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for model_name in MODELS:
        data = payload.get("model_summaries", {}).get(model_name)
        if not data:
            continue
        candidates = [
            row
            for row in data.get("by_component", [])
            if (row.get("mean_single_net_delta") is not None and row.get("mean_single_net_delta") < 0)
            or (row.get("mean_loo_net_loss_vs_full") is not None and row.get("mean_loo_net_loss_vs_full") > 0)
        ]
        candidates = sorted(
            candidates,
            key=lambda r: (
                r.get("mean_loo_net_loss_vs_full") or -999999.0,
                -(r.get("mean_single_net_delta") or 999999.0),
            ),
            reverse=True,
        )
        for row in candidates[:16]:
            label = f"{row.get('component_kind')}:L{row.get('layer')}"
            lines.append(
                f"| {model_name} | `{label}` | {fmt(row.get('mean_single_net_delta'))} | "
                f"{fmt(row.get('mean_single_emerged'))} | {fmt(row.get('mean_single_bias_delta'))} | "
                f"{fmt(row.get('mean_loo_net_loss_vs_full'))} | "
                f"`{json.dumps(row.get('label_counts') or {}, ensure_ascii=False)}` |"
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def summarize_round(round_name: str) -> dict[str, Any]:
    out_dir = RESULT_ROOT / round_name
    payload: dict[str, Any] = {"phase": 808, "round": round_name, "status": "missing", "model_summaries": {}, "models": []}
    for model_name in MODELS:
        path = out_dir / f"phase808_{model_name}_summary.json"
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        payload["model_summaries"][model_name] = data
        payload["models"].append(model_name)
    payload["status"] = "complete" if len(payload["models"]) == len(MODELS) else "partial"
    write_json(out_dir / "phase808_cross_model_summary.json", payload)
    write_markdown(out_dir / "phase808_cross_model_summary.md", payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = p807.build_parser()
    parser.add_argument("--format-beta", type=float, default=1.0)
    parser.add_argument("--identity-beta", type=float, default=0.0)
    parser.add_argument("--max-emergence-rate-for-source", type=float, default=0.15)
    parser.add_argument("--min-loo-net-loss", type=float, default=3.0)
    parser.add_argument("--min-loo-bias-loss", type=float, default=0.05)
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
