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
import phase801_target_neutral_suppressor_causal_test as p801  # noqa: E402
import phase802_new_blocker_stabilization_dose_response as p802  # noqa: E402
import phase803_semantic_new_blocker_source_localization as p803  # noqa: E402
import phase804_true_semantic_suppressor_projection_search as p804  # noqa: E402
import phase805_residual_closure_blocker_audit as p805  # noqa: E402
import phase806_format_echo_identity_residual_suppressor_search as p806  # noqa: E402
from model_utils import release_model  # noqa: E402
from phase722_functional_head_atlas_causal_ablation import write_json, write_jsonl  # noqa: E402
from phase735_source_restricted_writer_validation import MODELS  # noqa: E402
from phase771_matched_causal_intervention_reliability_test import case_map_for  # noqa: E402
from phase773_instruction_source_disentanglement import fmt  # noqa: E402
from phase780_surface_form_component_localization import lm_head_weight  # noqa: E402
from phase795_multi_component_causal_fiber_closure import selected_route_components  # noqa: E402


RESULT_ROOT = Path("tests/result/phase807_readout_geometry_new_blocker_emergence_audit")


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


def merge_counter_dicts(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    return p805.merge_counter_dicts(rows, key)


def above_mask(logits: torch.Tensor, target_id: int) -> torch.Tensor:
    vals = logits.detach().float().cpu()
    target = float(vals[int(target_id)].item())
    mask = vals > target
    mask[int(target_id)] = False
    return mask


def top_examples_for_ids(
    args: argparse.Namespace,
    token_ids: torch.Tensor,
    base_vals: torch.Tensor,
    after_vals: torch.Tensor,
    target_id: int,
    contrast_id: int,
    prompt_ids: list[int],
    prompt_text: str,
    candidate_ids: set[int],
    case_values: set[str],
    sort_values: torch.Tensor,
    limit: int,
) -> tuple[dict[str, int], dict[str, list[float]], dict[str, list[float]], dict[str, dict[str, Any]]]:
    if token_ids.numel() == 0:
        return {}, {}, {}, {}
    order = torch.argsort(sort_values, descending=True)
    if limit > 0:
        order = order[:limit]
    base_target = float(base_vals[int(target_id)].item())
    after_target = float(after_vals[int(target_id)].item())
    counts: Counter[str] = Counter()
    gap_deltas: dict[str, list[float]] = defaultdict(list)
    token_deltas: dict[str, list[float]] = defaultdict(list)
    examples: dict[str, dict[str, Any]] = {}
    for idx_tensor in order.tolist():
        tid = int(token_ids[int(idx_tensor)].item())
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
        base_gap = float(base_vals[tid].item() - base_target)
        after_gap = float(after_vals[tid].item() - after_target)
        token_delta = float(after_vals[tid].item() - base_vals[tid].item())
        gap_delta = after_gap - base_gap
        counts[cls] += 1
        gap_deltas[cls].append(gap_delta)
        token_deltas[cls].append(token_delta)
        if cls not in examples:
            examples[cls] = {
                "token_id": tid,
                "token_text": text,
                "base_gap": base_gap,
                "after_gap": after_gap,
                "gap_delta": gap_delta,
                "token_logit_delta": token_delta,
            }
    return dict(counts), gap_deltas, token_deltas, examples


def transition_metrics(
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
    base_vals = base_logits.detach().float().cpu()
    after_vals = after_logits.detach().float().cpu()
    base_target = float(base_vals[int(target_id)].item())
    after_target = float(after_vals[int(target_id)].item())
    base = above_mask(base_logits, target_id)
    after = above_mask(after_logits, target_id)
    resolved_mask = base & ~after
    persistent_mask = base & after
    emerged_mask = after & ~base
    base_ids = torch.nonzero(base, as_tuple=False).flatten()
    after_ids = torch.nonzero(after, as_tuple=False).flatten()
    resolved_ids = torch.nonzero(resolved_mask, as_tuple=False).flatten()
    persistent_ids = torch.nonzero(persistent_mask, as_tuple=False).flatten()
    emerged_ids = torch.nonzero(emerged_mask, as_tuple=False).flatten()
    base_count = int(base_ids.numel())
    after_count = int(after_ids.numel())
    resolved_count = int(resolved_ids.numel())
    persistent_count = int(persistent_ids.numel())
    emerged_count = int(emerged_ids.numel())
    target_delta = after_target - base_target

    if base_ids.numel():
        base_gap = base_vals[base_ids] - base_target
        after_gap = after_vals[base_ids] - after_target
        base_token_suppression = base_vals[base_ids] - after_vals[base_ids]
        base_gap_reduction = base_gap - after_gap
    else:
        base_token_suppression = torch.empty(0)
        base_gap_reduction = torch.empty(0)
    if emerged_ids.numel():
        emerged_gap = after_vals[emerged_ids] - after_target
        emerged_sort = emerged_gap
        emerged_token_delta = after_vals[emerged_ids] - base_vals[emerged_ids]
        emerged_gap_delta = (after_vals[emerged_ids] - after_target) - (base_vals[emerged_ids] - base_target)
    else:
        emerged_gap = emerged_token_delta = emerged_gap_delta = emerged_sort = torch.empty(0)
    if resolved_ids.numel():
        resolved_base_gap = base_vals[resolved_ids] - base_target
        resolved_sort = resolved_base_gap
        resolved_token_suppression = base_vals[resolved_ids] - after_vals[resolved_ids]
        resolved_gap_reduction = (base_vals[resolved_ids] - base_target) - (after_vals[resolved_ids] - after_target)
    else:
        resolved_token_suppression = resolved_gap_reduction = resolved_sort = torch.empty(0)
    if persistent_ids.numel():
        persistent_after_gap = after_vals[persistent_ids] - after_target
        persistent_sort = persistent_after_gap
        persistent_token_suppression = base_vals[persistent_ids] - after_vals[persistent_ids]
        persistent_gap_reduction = (base_vals[persistent_ids] - base_target) - (after_vals[persistent_ids] - after_target)
    else:
        persistent_token_suppression = persistent_gap_reduction = persistent_sort = torch.empty(0)

    classify_limit = int(args.max_transition_classify)
    emerged_counts, emerged_gap_delta_by_cls, emerged_token_delta_by_cls, emerged_examples = top_examples_for_ids(
        args,
        emerged_ids,
        base_vals,
        after_vals,
        target_id,
        contrast_id,
        prompt_ids,
        prompt_text,
        candidate_ids,
        case_values,
        emerged_sort,
        classify_limit,
    )
    resolved_counts, resolved_gap_delta_by_cls, resolved_token_delta_by_cls, resolved_examples = top_examples_for_ids(
        args,
        resolved_ids,
        base_vals,
        after_vals,
        target_id,
        contrast_id,
        prompt_ids,
        prompt_text,
        candidate_ids,
        case_values,
        resolved_sort,
        classify_limit,
    )
    persistent_counts, persistent_gap_delta_by_cls, persistent_token_delta_by_cls, persistent_examples = top_examples_for_ids(
        args,
        persistent_ids,
        base_vals,
        after_vals,
        target_id,
        contrast_id,
        prompt_ids,
        prompt_text,
        candidate_ids,
        case_values,
        persistent_sort,
        classify_limit,
    )

    def mean_tensor(t: torch.Tensor) -> float | None:
        return float(t.mean().item()) if t.numel() else None

    def mean_nested(values: dict[str, list[float]]) -> dict[str, float | None]:
        return {k: safe_mean(v) for k, v in values.items()}

    return {
        "semantic_base_above_count": base_count,
        "after_above_count": after_count,
        "transition_resolved_count": resolved_count,
        "transition_persistent_count": persistent_count,
        "transition_emerged_count": emerged_count,
        "transition_net_count_delta": after_count - base_count,
        "transition_resolved_rate_vs_base": resolved_count / max(base_count, 1),
        "transition_emergence_rate_vs_base": emerged_count / max(base_count, 1),
        "transition_emergence_share_after": emerged_count / max(after_count, 1),
        "transition_target_logit_delta_vs_semantic_base": target_delta,
        "transition_mean_base_blocker_token_suppression": mean_tensor(base_token_suppression),
        "transition_mean_base_blocker_gap_reduction": mean_tensor(base_gap_reduction),
        "transition_mean_resolved_token_suppression": mean_tensor(resolved_token_suppression),
        "transition_mean_resolved_gap_reduction": mean_tensor(resolved_gap_reduction),
        "transition_mean_persistent_token_suppression": mean_tensor(persistent_token_suppression),
        "transition_mean_persistent_gap_reduction": mean_tensor(persistent_gap_reduction),
        "transition_mean_emerged_token_delta": mean_tensor(emerged_token_delta),
        "transition_mean_emerged_gap_delta": mean_tensor(emerged_gap_delta),
        "transition_mean_emerged_gap_after": mean_tensor(emerged_gap),
        "transition_emerged_class_counts": emerged_counts,
        "transition_resolved_class_counts": resolved_counts,
        "transition_persistent_class_counts": persistent_counts,
        "transition_emerged_class_mean_gap_delta": mean_nested(emerged_gap_delta_by_cls),
        "transition_emerged_class_mean_token_delta": mean_nested(emerged_token_delta_by_cls),
        "transition_resolved_class_mean_gap_delta": mean_nested(resolved_gap_delta_by_cls),
        "transition_resolved_class_mean_token_delta": mean_nested(resolved_token_delta_by_cls),
        "transition_persistent_class_mean_gap_delta": mean_nested(persistent_gap_delta_by_cls),
        "transition_persistent_class_mean_token_delta": mean_nested(persistent_token_delta_by_cls),
        "transition_emerged_top_examples": emerged_examples,
        "transition_resolved_top_examples": resolved_examples,
        "transition_persistent_top_examples": persistent_examples,
    }


def label_phase807(row: dict[str, Any], args: argparse.Namespace) -> str:
    if row.get("token_closure_gain"):
        return "token_closure"
    net = safe_float(row.get("transition_net_count_delta")) or 0.0
    emerged = safe_float(row.get("transition_emerged_count")) or 0.0
    resolved = safe_float(row.get("transition_resolved_count")) or 0.0
    emergence_rate = safe_float(row.get("transition_emergence_rate_vs_base")) or 0.0
    bias_delta = safe_float(row.get("residual_required_bias_delta_vs_semantic_base")) or 0.0
    fmt_supp = safe_float(row.get("format_echo_mean_suppression_vs_semantic_base")) or 0.0
    if net < 0 and emergence_rate <= float(args.max_emergence_rate_for_closer) and bias_delta <= 0:
        return "readout_closer_candidate_no_closure"
    if net < 0 and emerged <= resolved:
        return "net_blocker_reduction_with_emergence"
    if fmt_supp > float(args.min_local_suppression) and net > 0:
        return "local_suppression_global_field_deformation"
    if emerged > resolved:
        return "new_blocker_emergence_dominant"
    if bias_delta < 0 and net >= 0:
        return "bias_reduced_but_blocker_count_expands"
    return "mixed_transition"


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
                "row_kind": "phase807_error",
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
        semantic_projected, _semantic_metrics = p804.projected_state_target_semantic(
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
                    "row_kind": "phase807_error",
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
        semantic_base_snapshot = p805.residual_snapshot(
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
        for format_beta in parse_float_grid(args.format_beta_grid, [0.0, 1.0]):
            for identity_beta in parse_float_grid(args.identity_beta_grid, [0.0, 1.0]):
                projected, projection_metrics = p806.projected_state_multi(
                    recipient_state,
                    donor_state,
                    route_components,
                    target_direction,
                    semantic_direction,
                    format_direction,
                    identity_direction,
                    target_alpha,
                    float(args.semantic_beta),
                    format_beta,
                    identity_beta,
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
                    f"target_a{p802.alpha_label(target_alpha)}_sem_b{p802.alpha_label(float(args.semantic_beta))}_fmt_b{p802.alpha_label(format_beta)}_id_b{p802.alpha_label(identity_beta)}",
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
                row.update(
                    {
                        "row_kind": "phase807_readout_geometry_new_blocker_emergence_audit",
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
                        "target_direction_alpha": float(target_alpha),
                        "semantic_suppression_beta": float(args.semantic_beta),
                        "format_echo_suppression_beta": float(format_beta),
                        "identity_anchor_suppression_beta": float(identity_beta),
                        "semantic_direction_mode": args.semantic_direction_mode,
                        "alpha0_semantic_new_blocker_count": len(alpha0_semantic),
                        "semantic_base_residual_full_above_count": semantic_base_snapshot.get("residual_full_above_count"),
                        "semantic_base_residual_required_bias_to_clear_all": semantic_base_snapshot.get("residual_required_bias_to_clear_all"),
                        "semantic_base_residual_semantic_share": semantic_base_snapshot.get("residual_semantic_share"),
                        "semantic_base_residual_format_echo_share": semantic_base_snapshot.get("residual_format_echo_share"),
                        "semantic_base_identity_anchor_fragmented": semantic_base_snapshot.get("residual_identity_anchor_fragmented"),
                        "format_echo_direction_token_ids": direction_tokens["format_echo_ids"],
                        "identity_anchor_direction_token_ids": direction_tokens["identity_variant_ids"],
                        "direction_class_counts": direction_tokens["direction_class_counts"],
                    }
                )
                if after_logits.numel():
                    row.update(
                        p805.residual_snapshot(
                            args,
                            after_logits,
                            target_id,
                            contrast_id,
                            recipient_ids,
                            recipient_prompt,
                            recipient_candidate_ids,
                            case_vals,
                            str(case.get("answer", "")),
                        )
                    )
                    row.update(
                        p806.tracked_token_metrics(
                            semantic_logits,
                            after_logits,
                            target_id,
                            direction_tokens["format_echo_ids"],
                            "format_echo",
                        )
                    )
                    row.update(
                        p806.tracked_token_metrics(
                            semantic_logits,
                            after_logits,
                            target_id,
                            direction_tokens["identity_variant_ids"],
                            "identity_anchor",
                        )
                    )
                    row.update(
                        transition_metrics(
                            args,
                            semantic_logits,
                            after_logits,
                            target_id,
                            contrast_id,
                            recipient_ids,
                            recipient_prompt,
                            recipient_candidate_ids,
                            case_vals,
                        )
                    )
                    row["residual_full_above_delta_vs_semantic_base"] = (
                        (safe_float(row.get("residual_full_above_count")) or 0.0)
                        - (safe_float(semantic_base_snapshot.get("residual_full_above_count")) or 0.0)
                    )
                    row["residual_required_bias_delta_vs_semantic_base"] = (
                        (safe_float(row.get("residual_required_bias_to_clear_all")) or 0.0)
                        - (safe_float(semantic_base_snapshot.get("residual_required_bias_to_clear_all")) or 0.0)
                    )
                    row["phase807_label"] = label_phase807(row, args)
                    row["phase807_boundary"] = (
                        "This phase audits blocker-set transitions from semantic-only baseline to residual projections. "
                        "It separates resolved old blockers from emerged new blockers, but does not yet localize a neuron-level closer."
                    )
                rows.append(row)
    return rows


def group_rows(rows: list[dict[str, Any]], fields: list[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("row_kind") == "phase807_readout_geometry_new_blocker_emergence_audit":
            groups[tuple(row.get(f) for f in fields)].append(row)
    out: list[dict[str, Any]] = []
    for key, vals in groups.items():
        payload = {field: value for field, value in zip(fields, key)}
        payload.update(
            {
                "n": len(vals),
                "case_n": len({v.get("case_id") for v in vals}),
                "mean_semantic_base_above_count": safe_mean([v.get("semantic_base_above_count") for v in vals]),
                "mean_after_above_count": safe_mean([v.get("after_above_count") for v in vals]),
                "mean_net_count_delta": safe_mean([v.get("transition_net_count_delta") for v in vals]),
                "mean_resolved_count": safe_mean([v.get("transition_resolved_count") for v in vals]),
                "mean_persistent_count": safe_mean([v.get("transition_persistent_count") for v in vals]),
                "mean_emerged_count": safe_mean([v.get("transition_emerged_count") for v in vals]),
                "mean_resolved_rate_vs_base": safe_mean([v.get("transition_resolved_rate_vs_base") for v in vals]),
                "mean_emergence_rate_vs_base": safe_mean([v.get("transition_emergence_rate_vs_base") for v in vals]),
                "mean_emergence_share_after": safe_mean([v.get("transition_emergence_share_after") for v in vals]),
                "mean_target_logit_delta_vs_semantic_base": safe_mean([v.get("transition_target_logit_delta_vs_semantic_base") for v in vals]),
                "mean_base_blocker_token_suppression": safe_mean([v.get("transition_mean_base_blocker_token_suppression") for v in vals]),
                "mean_base_blocker_gap_reduction": safe_mean([v.get("transition_mean_base_blocker_gap_reduction") for v in vals]),
                "mean_emerged_token_delta": safe_mean([v.get("transition_mean_emerged_token_delta") for v in vals]),
                "mean_emerged_gap_delta": safe_mean([v.get("transition_mean_emerged_gap_delta") for v in vals]),
                "mean_residual_required_bias_delta_vs_semantic_base": safe_mean([v.get("residual_required_bias_delta_vs_semantic_base") for v in vals]),
                "mean_format_echo_suppression_vs_semantic_base": safe_mean([v.get("format_echo_mean_suppression_vs_semantic_base") for v in vals]),
                "mean_identity_suppression_vs_semantic_base": safe_mean([v.get("identity_anchor_mean_suppression_vs_semantic_base") for v in vals]),
                "identity_anchor_fragmented_rate": safe_rate([v.get("residual_identity_anchor_fragmented") for v in vals]),
                "token_closure_gain_rate": safe_rate([v.get("token_closure_gain") for v in vals]),
                "label_counts": dict(Counter(v.get("phase807_label") for v in vals)),
                "emerged_class_counts": merge_counter_dicts(vals, "transition_emerged_class_counts"),
                "resolved_class_counts": merge_counter_dicts(vals, "transition_resolved_class_counts"),
                "persistent_class_counts": merge_counter_dicts(vals, "transition_persistent_class_counts"),
                "residual_class_counts": merge_counter_dicts(vals, "residual_class_counts"),
            }
        )
        out.append(payload)
    out.sort(
        key=lambda r: (
            r.get("token_closure_gain_rate") or 0.0,
            -(r.get("mean_net_count_delta") or 999999.0),
            -(r.get("mean_emerged_count") or 999999.0),
        ),
        reverse=True,
    )
    return out


def summarize(rows: list[dict[str, Any]], args: argparse.Namespace, attn_impl: str, routes: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "phase": 807,
        "title": "Readout Geometry and New-Blocker Emergence Audit",
        "model": args.model,
        "round": args.round_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "attn_implementation": attn_impl,
        "dtype": "bfloat16",
        "quantization": "off",
        "n_rows": len(rows),
        "n_cases": len({r.get("case_id") for r in rows if r.get("row_kind") == "phase807_readout_geometry_new_blocker_emergence_audit"}),
        "n_routes": len(routes),
        "target_alpha_grid": parse_float_grid(args.target_alpha_grid, [0.75]),
        "semantic_beta": float(args.semantic_beta),
        "format_beta_grid": parse_float_grid(args.format_beta_grid, [0.0, 1.0]),
        "identity_beta_grid": parse_float_grid(args.identity_beta_grid, [0.0, 1.0]),
        "by_projection": group_rows(
            rows,
            [
                "model",
                "target_direction_alpha",
                "semantic_suppression_beta",
                "format_echo_suppression_beta",
                "identity_anchor_suppression_beta",
            ],
        ),
        "by_route_projection": group_rows(
            rows,
            [
                "model",
                "route_component_signature",
                "target_direction_alpha",
                "format_echo_suppression_beta",
                "identity_anchor_suppression_beta",
            ],
        )[:160],
        "by_case_projection": group_rows(
            rows,
            [
                "model",
                "case_id",
                "target_direction_alpha",
                "format_echo_suppression_beta",
                "identity_anchor_suppression_beta",
            ],
        )[:160],
        "strict_boundary": (
            "This phase compares semantic-only and residual-projected blocker sets. It diagnoses readout field deformation "
            "and new-blocker emergence; it is not a proof of token closure."
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
        f"format_beta={args.format_beta_grid} identity_beta={args.identity_beta_grid}"
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
                log(f"{args.model}: readout geometry/new blocker emergence audit {ci}/{len(selected)} cases; rows={len(rows)}")
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
    write_jsonl(out_dir / f"phase807_{args.model}_rows.jsonl", rows)
    write_json(out_dir / f"phase807_{args.model}_summary.json", summary)
    print(
        json.dumps(
            {
                "model": args.model,
                "round": args.round_name,
                "attn": attn_impl,
                "n_cases": summary["n_cases"],
                "n_rows": summary["n_rows"],
                "by_projection": summary["by_projection"][:12],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return summary


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        f"# Phase 807 Readout Geometry and New-Blocker Emergence Audit ({payload['round']})",
        "",
        f"- Status: `{payload['status']}`",
        "- Boundary: compares semantic-only blocker sets against residual projections.",
        "- It separates resolved old blockers from emerged new blockers; token closure remains a separate criterion.",
        "",
        "## By Projection",
        "",
        "| model | fmt beta | id beta | rows | cases | base | after | net | resolved | emerged | emergence rate | emergence share | target delta | base supp | gap red | new token delta | new gap delta | bias delta | fmt supp | id supp | anchor frag | closure | labels |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for model_name in MODELS:
        data = payload.get("model_summaries", {}).get(model_name)
        if not data:
            continue
        rows = sorted(
            data.get("by_projection", []),
            key=lambda r: (
                safe_float(r.get("format_echo_suppression_beta")) or 0.0,
                safe_float(r.get("identity_anchor_suppression_beta")) or 0.0,
            ),
        )
        for row in rows:
            lines.append(
                f"| {model_name} | {fmt(row.get('format_echo_suppression_beta'))} | {fmt(row.get('identity_anchor_suppression_beta'))} | "
                f"{row.get('n')} | {row.get('case_n')} | {fmt(row.get('mean_semantic_base_above_count'))} | "
                f"{fmt(row.get('mean_after_above_count'))} | {fmt(row.get('mean_net_count_delta'))} | "
                f"{fmt(row.get('mean_resolved_count'))} | {fmt(row.get('mean_emerged_count'))} | "
                f"{fmt(row.get('mean_emergence_rate_vs_base'))} | {fmt(row.get('mean_emergence_share_after'))} | "
                f"{fmt(row.get('mean_target_logit_delta_vs_semantic_base'))} | {fmt(row.get('mean_base_blocker_token_suppression'))} | "
                f"{fmt(row.get('mean_base_blocker_gap_reduction'))} | {fmt(row.get('mean_emerged_token_delta'))} | "
                f"{fmt(row.get('mean_emerged_gap_delta'))} | {fmt(row.get('mean_residual_required_bias_delta_vs_semantic_base'))} | "
                f"{fmt(row.get('mean_format_echo_suppression_vs_semantic_base'))} | {fmt(row.get('mean_identity_suppression_vs_semantic_base'))} | "
                f"{fmt(row.get('identity_anchor_fragmented_rate'))} | {fmt(row.get('token_closure_gain_rate'))} | "
                f"`{json.dumps(row.get('label_counts') or {}, ensure_ascii=False)}` |"
            )
    lines += [
        "",
        "## Emerged Class Counts",
        "",
        "| model | fmt beta | id beta | class | count |",
        "|---|---:|---:|---|---:|",
    ]
    for model_name in MODELS:
        data = payload.get("model_summaries", {}).get(model_name)
        if not data:
            continue
        for row in data.get("by_projection", []):
            for cls, count in sorted((row.get("emerged_class_counts") or {}).items(), key=lambda kv: (-kv[1], kv[0])):
                lines.append(
                    f"| {model_name} | {fmt(row.get('format_echo_suppression_beta'))} | "
                    f"{fmt(row.get('identity_anchor_suppression_beta'))} | `{cls}` | {count} |"
                )
    lines += [
        "",
        "## Resolved Class Counts",
        "",
        "| model | fmt beta | id beta | class | count |",
        "|---|---:|---:|---|---:|",
    ]
    for model_name in MODELS:
        data = payload.get("model_summaries", {}).get(model_name)
        if not data:
            continue
        for row in data.get("by_projection", []):
            for cls, count in sorted((row.get("resolved_class_counts") or {}).items(), key=lambda kv: (-kv[1], kv[0])):
                lines.append(
                    f"| {model_name} | {fmt(row.get('format_echo_suppression_beta'))} | "
                    f"{fmt(row.get('identity_anchor_suppression_beta'))} | `{cls}` | {count} |"
                )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def summarize_round(round_name: str) -> dict[str, Any]:
    out_dir = RESULT_ROOT / round_name
    payload: dict[str, Any] = {"phase": 807, "round": round_name, "status": "missing", "model_summaries": {}, "models": []}
    for model_name in MODELS:
        path = out_dir / f"phase807_{model_name}_summary.json"
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        payload["model_summaries"][model_name] = data
        payload["models"].append(model_name)
    payload["status"] = "complete" if len(payload["models"]) == len(MODELS) else "partial"
    write_json(out_dir / "phase807_cross_model_summary.json", payload)
    write_markdown(out_dir / "phase807_cross_model_summary.md", payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = p806.build_parser()
    parser.add_argument("--max-transition-classify", type=int, default=8000)
    parser.add_argument("--max-emergence-rate-for-closer", type=float, default=0.15)
    parser.add_argument("--min-local-suppression", type=float, default=0.25)
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
