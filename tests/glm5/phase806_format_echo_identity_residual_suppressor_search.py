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
from model_utils import release_model  # noqa: E402
from phase722_functional_head_atlas_causal_ablation import write_json, write_jsonl  # noqa: E402
from phase735_source_restricted_writer_validation import MODELS  # noqa: E402
from phase771_matched_causal_intervention_reliability_test import case_map_for  # noqa: E402
from phase773_instruction_source_disentanglement import fmt  # noqa: E402
from phase780_surface_form_component_localization import lm_head_weight  # noqa: E402
from phase795_multi_component_causal_fiber_closure import selected_route_components  # noqa: E402


RESULT_ROOT = Path("tests/result/phase806_format_echo_identity_residual_suppressor_search")
FORMAT_ECHO_CLASSES = p805.FORMAT_ECHO_CLASSES


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def parse_float_grid(text: str, fallback: list[float]) -> list[float]:
    return p805.parse_float_grid(text, fallback)


def safe_float(value: Any) -> float | None:
    return p805.safe_float(value)


def safe_mean(values: list[Any]) -> float | None:
    return p805.safe_mean(values)


def safe_rate(values: list[Any]) -> float | None:
    return p805.safe_rate(values)


def merge_counter_dicts(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    return p805.merge_counter_dicts(rows, key)


def orthogonalize_many(vec: torch.Tensor, bases: list[torch.Tensor]) -> torch.Tensor:
    out = vec.float()
    for base in bases:
        b = base.float()
        denom = float(torch.dot(b, b).item())
        if denom > 1e-9:
            out = out - float(torch.dot(out, b).item() / denom) * b
    return out


def direction_from_ids(unembed: torch.Tensor, target_id: int, ids: list[int]) -> torch.Tensor:
    if not ids:
        return torch.zeros_like(unembed[int(target_id)].float().cpu())
    uniq = sorted({int(tid) for tid in ids if int(tid) != int(target_id)})
    if not uniq:
        return torch.zeros_like(unembed[int(target_id)].float().cpu())
    mean_vec = unembed[torch.tensor(uniq, dtype=torch.long)].float().mean(dim=0).cpu()
    target = unembed[int(target_id)].float().cpu()
    return mean_vec - target


def collect_residual_direction_tokens(
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
    vals = logits.detach().float().cpu()
    target_logit = float(vals[int(target_id)].item())
    above_mask = vals > target_logit
    above_mask[int(target_id)] = False
    above_ids = torch.nonzero(above_mask, as_tuple=False).flatten()
    if above_ids.numel() == 0:
        return {
            "format_echo_ids": [],
            "identity_variant_ids": [],
            "semantic_ids": [],
            "direction_class_counts": {},
            "direction_examples": {},
            "direction_scan_count": 0,
        }
    above_vals = vals[above_ids]
    _sorted_vals, order = torch.sort(above_vals, descending=True)
    sorted_ids = above_ids[order]
    limit = min(int(sorted_ids.numel()), int(args.max_residual_direction_scan))
    target_norm = p798.norm(target_answer)
    format_echo_ids: list[int] = []
    identity_ids: list[int] = []
    semantic_ids: list[int] = []
    counts: Counter[str] = Counter()
    examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for offset in range(limit):
        tid = int(sorted_ids[offset].item())
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
        counts[cls] += 1
        if len(examples[cls]) < int(args.max_direction_examples_saved):
            examples[cls].append(
                {
                    "token_id": tid,
                    "token_text": text,
                    "gap_above_target": float(vals[tid].item() - target_logit),
                    "global_rank": offset + 1,
                }
            )
        if cls in FORMAT_ECHO_CLASSES and len(format_echo_ids) < int(args.max_class_direction_tokens):
            format_echo_ids.append(tid)
        if cls == "semantic_or_lexical_competitor" and len(semantic_ids) < int(args.max_class_direction_tokens):
            semantic_ids.append(tid)
        if target_norm and p798.norm(text) == target_norm and tid != int(target_id):
            if len(identity_ids) < int(args.max_identity_direction_tokens):
                identity_ids.append(tid)
    return {
        "format_echo_ids": format_echo_ids,
        "identity_variant_ids": identity_ids,
        "semantic_ids": semantic_ids,
        "direction_class_counts": dict(counts),
        "direction_examples": dict(examples),
        "direction_scan_count": limit,
    }


def remove_direction(
    delta: torch.Tensor,
    direction: torch.Tensor,
    beta: float,
    bases: list[torch.Tensor],
    prefix: str,
) -> tuple[torch.Tensor, dict[str, Any], torch.Tensor]:
    basis = orthogonalize_many(direction.float(), bases)
    denom = float(torch.dot(basis, basis).item())
    if denom <= 1e-9 or float(beta) == 0.0:
        parallel = torch.zeros_like(delta.float())
        projected = delta.float()
    else:
        parallel = float(torch.dot(delta.float(), basis).item() / denom) * basis
        projected = delta.float() - float(beta) * parallel
    meta = {
        f"{prefix}_beta": float(beta),
        f"{prefix}_basis_norm": float(basis.norm().item()),
        f"{prefix}_component_before": float(torch.dot(delta.float(), basis).item()),
        f"{prefix}_component_after": float(torch.dot(projected.float(), basis).item()),
        f"{prefix}_parallel_norm": float(parallel.norm().item()),
    }
    return projected, meta, basis


def project_delta_target_semantic_format_identity(
    delta: torch.Tensor,
    target_direction: torch.Tensor,
    semantic_direction: torch.Tensor,
    format_direction: torch.Tensor,
    identity_direction: torch.Tensor,
    target_alpha: float,
    semantic_beta: float,
    format_beta: float,
    identity_beta: float,
) -> tuple[torch.Tensor, dict[str, Any]]:
    projected, meta = p804.project_delta_target_semantic(
        delta,
        target_direction,
        semantic_direction,
        target_alpha,
        semantic_beta,
    )
    target_basis = target_direction.float()
    semantic_basis = orthogonalize_many(semantic_direction.float(), [target_basis])
    projected, fmt_meta, format_basis = remove_direction(
        projected,
        format_direction,
        format_beta,
        [target_basis, semantic_basis],
        "format_echo",
    )
    projected, id_meta, _identity_basis = remove_direction(
        projected,
        identity_direction,
        identity_beta,
        [target_basis, semantic_basis, format_basis],
        "identity_anchor",
    )
    meta.update(fmt_meta)
    meta.update(id_meta)
    meta["projected_norm_after_residual_suppressors"] = float(projected.float().norm().item())
    return projected, meta


def projected_state_multi(
    recipient_state: dict[str, Any],
    donor_state: dict[str, Any],
    route_components: list[dict[str, Any]],
    target_direction: torch.Tensor,
    semantic_direction: torch.Tensor,
    format_direction: torch.Tensor,
    identity_direction: torch.Tensor,
    target_alpha: float,
    semantic_beta: float,
    format_beta: float,
    identity_beta: float,
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
        delta_projected, meta = project_delta_target_semantic_format_identity(
            delta,
            target_direction,
            semantic_direction,
            format_direction,
            identity_direction,
            target_alpha,
            semantic_beta,
            format_beta,
            identity_beta,
        )
        projected[key] = (rec_vec.float() + delta_projected).detach().cpu()
        for k, v in meta.items():
            metrics[k].append(v)
    summary = {f"mean_{k}": safe_mean(vals) for k, vals in metrics.items()}
    summary["target_direction_alpha"] = float(target_alpha)
    summary["semantic_suppression_beta"] = float(semantic_beta)
    summary["format_echo_suppression_beta"] = float(format_beta)
    summary["identity_anchor_suppression_beta"] = float(identity_beta)
    return projected, summary


def tracked_token_metrics(
    base_logits: torch.Tensor,
    after_logits: torch.Tensor,
    target_id: int,
    ids: list[int],
    prefix: str,
) -> dict[str, Any]:
    uniq = sorted({int(tid) for tid in ids if int(tid) != int(target_id)})
    if not uniq or not after_logits.numel():
        return {
            f"{prefix}_direction_token_count": len(uniq),
            f"{prefix}_mean_suppression_vs_semantic_base": None,
            f"{prefix}_still_above_after_rate": None,
            f"{prefix}_mean_gap_after": None,
        }
    base_vals = base_logits.detach().float().cpu()
    after_vals = after_logits.detach().float().cpu()
    after_target = float(after_vals[int(target_id)].item())
    suppressions = [float(base_vals[tid].item() - after_vals[tid].item()) for tid in uniq]
    still = [bool(float(after_vals[tid].item()) > after_target) for tid in uniq]
    gaps = [float(after_vals[tid].item() - after_target) for tid in uniq]
    return {
        f"{prefix}_direction_token_count": len(uniq),
        f"{prefix}_mean_suppression_vs_semantic_base": safe_mean(suppressions),
        f"{prefix}_still_above_after_rate": safe_rate(still),
        f"{prefix}_mean_gap_after": safe_mean(gaps),
    }


def label_phase806(row: dict[str, Any], args: argparse.Namespace) -> str:
    if row.get("token_closure_gain"):
        return "token_closure"
    fmt_beta = safe_float(row.get("format_echo_suppression_beta")) or 0.0
    id_beta = safe_float(row.get("identity_anchor_suppression_beta")) or 0.0
    fmt_supp = safe_float(row.get("format_echo_mean_suppression_vs_semantic_base")) or 0.0
    id_supp = safe_float(row.get("identity_anchor_mean_suppression_vs_semantic_base")) or 0.0
    res_delta = safe_float(row.get("residual_full_above_delta_vs_semantic_base")) or 0.0
    fmt_share_delta = safe_float(row.get("residual_format_echo_share_delta_vs_semantic_base")) or 0.0
    id_count_delta = safe_float(row.get("surface_variant_count_delta_vs_semantic_base")) or 0.0
    if fmt_beta > 0 and id_beta > 0 and res_delta < 0 and (fmt_supp > 0 or id_supp > 0):
        return "combined_residual_direction_reduces_blockers"
    if fmt_beta > 0 and fmt_supp >= float(args.min_class_suppression) and fmt_share_delta <= 0:
        return "format_echo_direction_effective_no_closure"
    if id_beta > 0 and id_supp >= float(args.min_class_suppression) and id_count_delta <= 0:
        return "identity_direction_effective_no_closure"
    if res_delta < 0:
        return "residual_count_reduced_but_class_target_unclear"
    if fmt_beta > 0 or id_beta > 0:
        return "direction_projection_weak_or_backfires"
    return "semantic_only_baseline"


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
                "row_kind": "phase806_error",
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
                    "row_kind": "phase806_error",
                    "model": args.model,
                    "case_id": case["case_id"],
                    "route_id": route["route_id"],
                    "target_direction_alpha": float(target_alpha),
                    "semantic_suppression_beta": float(args.semantic_beta),
                    "error": semantic_error or "empty_semantic_base_logits",
                }
            )
            continue
        direction_tokens = collect_residual_direction_tokens(
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
        format_direction = direction_from_ids(unembed, target_id, direction_tokens["format_echo_ids"])
        identity_direction = direction_from_ids(unembed, target_id, direction_tokens["identity_variant_ids"])
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
                projected, projection_metrics = projected_state_multi(
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
                        "row_kind": "phase806_format_echo_identity_residual_suppressor_search",
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
                        "semantic_base_surface_target_variant_count_above": semantic_base_snapshot.get("residual_surface_target_variant_count_above"),
                        "semantic_base_identity_anchor_fragmented": semantic_base_snapshot.get("residual_identity_anchor_fragmented"),
                        "format_echo_direction_token_ids": direction_tokens["format_echo_ids"],
                        "identity_anchor_direction_token_ids": direction_tokens["identity_variant_ids"],
                        "semantic_residual_direction_token_ids": direction_tokens["semantic_ids"],
                        "direction_class_counts": direction_tokens["direction_class_counts"],
                        "direction_examples": direction_tokens["direction_examples"],
                        "direction_scan_count": direction_tokens["direction_scan_count"],
                    }
                )
                if after_logits.numel():
                    row.update(
                        p802.classify_new_blockers(
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
                    row.update(
                        p803.matched_semantic_metrics(
                            recipient_state["logits"],
                            alpha0_logits,
                            after_logits,
                            target_id,
                            alpha0_semantic,
                        )
                    )
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
                        tracked_token_metrics(
                            semantic_logits,
                            after_logits,
                            target_id,
                            direction_tokens["format_echo_ids"],
                            "format_echo",
                        )
                    )
                    row.update(
                        tracked_token_metrics(
                            semantic_logits,
                            after_logits,
                            target_id,
                            direction_tokens["identity_variant_ids"],
                            "identity_anchor",
                        )
                    )
                    row["target_gain_delta_vs_semantic_base"] = float(
                        after_logits[int(target_id)].item() - semantic_logits[int(target_id)].item()
                    )
                    row["residual_full_above_delta_vs_semantic_base"] = (
                        (safe_float(row.get("residual_full_above_count")) or 0.0)
                        - (safe_float(semantic_base_snapshot.get("residual_full_above_count")) or 0.0)
                    )
                    row["residual_required_bias_delta_vs_semantic_base"] = (
                        (safe_float(row.get("residual_required_bias_to_clear_all")) or 0.0)
                        - (safe_float(semantic_base_snapshot.get("residual_required_bias_to_clear_all")) or 0.0)
                    )
                    row["residual_format_echo_share_delta_vs_semantic_base"] = (
                        (safe_float(row.get("residual_format_echo_share")) or 0.0)
                        - (safe_float(semantic_base_snapshot.get("residual_format_echo_share")) or 0.0)
                    )
                    row["residual_semantic_share_delta_vs_semantic_base"] = (
                        (safe_float(row.get("residual_semantic_share")) or 0.0)
                        - (safe_float(semantic_base_snapshot.get("residual_semantic_share")) or 0.0)
                    )
                    row["surface_variant_count_delta_vs_semantic_base"] = (
                        (safe_float(row.get("residual_surface_target_variant_count_above")) or 0.0)
                        - (safe_float(semantic_base_snapshot.get("residual_surface_target_variant_count_above")) or 0.0)
                    )
                    row["phase806_label"] = label_phase806(row, args)
                    row["phase806_boundary"] = (
                        "This phase tests residual direction projections for format/echo and identity-anchor classes "
                        "after fixing semantic suppression. It is direction-level evidence, not neuron-level closure."
                    )
                rows.append(row)
    return rows


def group_rows(rows: list[dict[str, Any]], fields: list[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("row_kind") == "phase806_format_echo_identity_residual_suppressor_search":
            groups[tuple(row.get(f) for f in fields)].append(row)
    out: list[dict[str, Any]] = []
    for key, vals in groups.items():
        payload = {field: value for field, value in zip(fields, key)}
        payload.update(
            {
                "n": len(vals),
                "case_n": len({v.get("case_id") for v in vals}),
                "mean_target_gain_delta_vs_semantic_base": safe_mean([v.get("target_gain_delta_vs_semantic_base") for v in vals]),
                "mean_old_blocker_suppression": safe_mean([v.get("baseline_blocker_mean_suppression") for v in vals]),
                "mean_matched_semantic_suppression": safe_mean([v.get("matched_semantic_true_suppression_vs_alpha0") for v in vals]),
                "mean_matched_semantic_still_above": safe_mean([v.get("matched_semantic_still_above_target_rate") for v in vals]),
                "mean_semantic_base_residual_full_above_count": safe_mean([v.get("semantic_base_residual_full_above_count") for v in vals]),
                "mean_residual_full_above_count": safe_mean([v.get("residual_full_above_count") for v in vals]),
                "mean_residual_full_above_delta_vs_semantic_base": safe_mean([v.get("residual_full_above_delta_vs_semantic_base") for v in vals]),
                "mean_residual_required_bias_delta_vs_semantic_base": safe_mean([v.get("residual_required_bias_delta_vs_semantic_base") for v in vals]),
                "mean_residual_semantic_share": safe_mean([v.get("residual_semantic_share") for v in vals]),
                "mean_residual_format_echo_share": safe_mean([v.get("residual_format_echo_share") for v in vals]),
                "mean_residual_format_echo_share_delta_vs_semantic_base": safe_mean([v.get("residual_format_echo_share_delta_vs_semantic_base") for v in vals]),
                "mean_format_echo_direction_token_count": safe_mean([v.get("format_echo_direction_token_count") for v in vals]),
                "mean_format_echo_suppression_vs_semantic_base": safe_mean([v.get("format_echo_mean_suppression_vs_semantic_base") for v in vals]),
                "mean_format_echo_still_above_after_rate": safe_mean([v.get("format_echo_still_above_after_rate") for v in vals]),
                "mean_identity_direction_token_count": safe_mean([v.get("identity_anchor_direction_token_count") for v in vals]),
                "mean_identity_suppression_vs_semantic_base": safe_mean([v.get("identity_anchor_mean_suppression_vs_semantic_base") for v in vals]),
                "mean_identity_still_above_after_rate": safe_mean([v.get("identity_anchor_still_above_after_rate") for v in vals]),
                "mean_surface_variant_count_delta_vs_semantic_base": safe_mean([v.get("surface_variant_count_delta_vs_semantic_base") for v in vals]),
                "identity_anchor_fragmented_rate": safe_rate([v.get("residual_identity_anchor_fragmented") for v in vals]),
                "token_closure_gain_rate": safe_rate([v.get("token_closure_gain") for v in vals]),
                "label_counts": dict(Counter(v.get("phase806_label") for v in vals)),
                "residual_class_counts": merge_counter_dicts(vals, "residual_class_counts"),
                "direction_class_counts": merge_counter_dicts(vals, "direction_class_counts"),
            }
        )
        out.append(payload)
    out.sort(
        key=lambda r: (
            r.get("token_closure_gain_rate") or 0.0,
            -(r.get("mean_residual_full_above_count") or 999999.0),
            -(r.get("mean_residual_required_bias_delta_vs_semantic_base") or 999999.0),
        ),
        reverse=True,
    )
    return out


def summarize(rows: list[dict[str, Any]], args: argparse.Namespace, attn_impl: str, routes: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "phase": 806,
        "title": "Format/Echo and Identity-Anchor Residual Suppressor Search",
        "model": args.model,
        "round": args.round_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "attn_implementation": attn_impl,
        "dtype": "bfloat16",
        "quantization": "off",
        "n_rows": len(rows),
        "n_cases": len({r.get("case_id") for r in rows if r.get("row_kind") == "phase806_format_echo_identity_residual_suppressor_search"}),
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
            "This phase builds residual format/echo and identity-anchor directions from the semantic-suppressed blocker field. "
            "It tests direction-level suppression after semantic beta=1; it does not identify neuron-level suppressors."
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
                log(f"{args.model}: format/echo identity residual suppressor search {ci}/{len(selected)} cases; rows={len(rows)}")
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
    write_jsonl(out_dir / f"phase806_{args.model}_rows.jsonl", rows)
    write_json(out_dir / f"phase806_{args.model}_summary.json", summary)
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
        f"# Phase 806 Format/Echo and Identity-Anchor Residual Suppressor Search ({payload['round']})",
        "",
        f"- Status: `{payload['status']}`",
        "- Boundary: direction-level projection after semantic suppression, not neuron-level suppressor discovery.",
        "- Baseline for deltas is semantic-only projection with semantic beta fixed.",
        "",
        "## By Projection",
        "",
        "| model | target alpha | sem beta | fmt beta | id beta | rows | cases | base blockers | blockers | blocker delta | bias delta | fmt supp | fmt still | id supp | id still | fmt share | fmt share delta | anchor frag | closure | labels |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for model_name in MODELS:
        data = payload.get("model_summaries", {}).get(model_name)
        if not data:
            continue
        rows = sorted(
            data.get("by_projection", []),
            key=lambda r: (
                safe_float(r.get("target_direction_alpha")) or 0.0,
                safe_float(r.get("format_echo_suppression_beta")) or 0.0,
                safe_float(r.get("identity_anchor_suppression_beta")) or 0.0,
            ),
        )
        for row in rows:
            lines.append(
                f"| {model_name} | {fmt(row.get('target_direction_alpha'))} | {fmt(row.get('semantic_suppression_beta'))} | "
                f"{fmt(row.get('format_echo_suppression_beta'))} | {fmt(row.get('identity_anchor_suppression_beta'))} | "
                f"{row.get('n')} | {row.get('case_n')} | {fmt(row.get('mean_semantic_base_residual_full_above_count'))} | "
                f"{fmt(row.get('mean_residual_full_above_count'))} | {fmt(row.get('mean_residual_full_above_delta_vs_semantic_base'))} | "
                f"{fmt(row.get('mean_residual_required_bias_delta_vs_semantic_base'))} | "
                f"{fmt(row.get('mean_format_echo_suppression_vs_semantic_base'))} | {fmt(row.get('mean_format_echo_still_above_after_rate'))} | "
                f"{fmt(row.get('mean_identity_suppression_vs_semantic_base'))} | {fmt(row.get('mean_identity_still_above_after_rate'))} | "
                f"{fmt(row.get('mean_residual_format_echo_share'))} | {fmt(row.get('mean_residual_format_echo_share_delta_vs_semantic_base'))} | "
                f"{fmt(row.get('identity_anchor_fragmented_rate'))} | {fmt(row.get('token_closure_gain_rate'))} | "
                f"`{json.dumps(row.get('label_counts') or {}, ensure_ascii=False)}` |"
            )
    lines += [
        "",
        "## Direction Token Class Counts",
        "",
        "| model | target alpha | fmt beta | id beta | class | count |",
        "|---|---:|---:|---:|---|---:|",
    ]
    for model_name in MODELS:
        data = payload.get("model_summaries", {}).get(model_name)
        if not data:
            continue
        for row in data.get("by_projection", []):
            for cls, count in sorted((row.get("direction_class_counts") or {}).items(), key=lambda kv: (-kv[1], kv[0])):
                lines.append(
                    f"| {model_name} | {fmt(row.get('target_direction_alpha'))} | "
                    f"{fmt(row.get('format_echo_suppression_beta'))} | {fmt(row.get('identity_anchor_suppression_beta'))} | `{cls}` | {count} |"
                )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def summarize_round(round_name: str) -> dict[str, Any]:
    out_dir = RESULT_ROOT / round_name
    payload: dict[str, Any] = {"phase": 806, "round": round_name, "status": "missing", "model_summaries": {}, "models": []}
    for model_name in MODELS:
        path = out_dir / f"phase806_{model_name}_summary.json"
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        payload["model_summaries"][model_name] = data
        payload["models"].append(model_name)
    payload["status"] = "complete" if len(payload["models"]) == len(MODELS) else "partial"
    write_json(out_dir / "phase806_cross_model_summary.json", payload)
    write_markdown(out_dir / "phase806_cross_model_summary.md", payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = p805.build_parser()
    parser.add_argument("--format-beta-grid", default="0,1")
    parser.add_argument("--identity-beta-grid", default="0,1")
    parser.add_argument("--semantic-beta", type=float, default=1.0)
    parser.add_argument("--max-residual-direction-scan", type=int, default=8000)
    parser.add_argument("--max-class-direction-tokens", type=int, default=64)
    parser.add_argument("--max-identity-direction-tokens", type=int, default=64)
    parser.add_argument("--max-direction-examples-saved", type=int, default=4)
    parser.add_argument("--min-class-suppression", type=float, default=0.1)
    parser.set_defaults(target_alpha_grid="0.75", semantic_beta_grid="1")
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
