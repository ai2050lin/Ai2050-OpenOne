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
import phase808_readout_closer_source_localization as p808  # noqa: E402
from model_utils import get_layers, release_model  # noqa: E402
from phase722_functional_head_atlas_causal_ablation import write_json, write_jsonl  # noqa: E402
from phase735_source_restricted_writer_validation import MODELS  # noqa: E402
from phase771_matched_causal_intervention_reliability_test import case_map_for  # noqa: E402
from phase773_instruction_source_disentanglement import fmt  # noqa: E402
from phase780_surface_form_component_localization import lm_head_weight  # noqa: E402
from phase786_head_mlp_source_audit import infer_num_heads  # noqa: E402
from phase795_multi_component_causal_fiber_closure import selected_route_components  # noqa: E402


RESULT_ROOT = Path("tests/result/phase809_late_layer_head_channel_closer_decomposition")


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


def mean_unembed(unembed: torch.Tensor, ids: list[int], target_id: int) -> torch.Tensor:
    uniq = sorted({int(tid) for tid in ids if int(tid) != int(target_id)})
    if not uniq:
        return unembed[int(target_id)].float().cpu()
    return unembed[torch.tensor(uniq, dtype=torch.long)].float().mean(dim=0).cpu()


def exact_unit_format_delta(
    raw_delta: torch.Tensor,
    target_direction: torch.Tensor,
    semantic_direction: torch.Tensor,
    format_direction: torch.Tensor,
    identity_direction: torch.Tensor,
    target_alpha: float,
    semantic_beta: float,
    format_beta: float,
) -> torch.Tensor:
    sem_delta, _sem_meta = p804.project_delta_target_semantic(
        raw_delta,
        target_direction,
        semantic_direction,
        target_alpha,
        semantic_beta,
    )
    full_delta, _full_meta = p806.project_delta_target_semantic_format_identity(
        raw_delta,
        target_direction,
        semantic_direction,
        format_direction,
        identity_direction,
        target_alpha,
        semantic_beta,
        format_beta,
        0.0,
    )
    return (full_delta.float() - sem_delta.float()).detach().cpu()


def select_unit_ids(scores: torch.Tensor, max_units: int) -> list[int]:
    if scores.numel() == 0 or max_units <= 0:
        return []
    vals = scores.detach().float().cpu()
    selected: list[int] = []
    positive = torch.clamp(vals, min=0.0)
    pos_n = min(max_units, max(1, max_units // 2), int((positive > 0).sum().item()))
    if pos_n > 0:
        _top_vals, top_ids = torch.topk(positive, pos_n)
        selected.extend(int(x) for x in top_ids.tolist() if float(positive[int(x)].item()) > 0)
    remain = max_units - len(dict.fromkeys(selected))
    if remain > 0:
        _abs_vals, abs_ids = torch.topk(vals.abs(), min(remain + len(selected), int(vals.numel())))
        for idx in abs_ids.tolist():
            if int(idx) not in selected:
                selected.append(int(idx))
            if len(selected) >= max_units:
                break
    return list(dict.fromkeys(selected))[:max_units]


def attention_unit_candidates(
    model,
    key: tuple[str, int],
    recipient_state: dict[str, Any],
    donor_state: dict[str, Any],
    unembed: torch.Tensor,
    target_id: int,
    format_ids: list[int],
    target_direction: torch.Tensor,
    semantic_direction: torch.Tensor,
    format_direction: torch.Tensor,
    identity_direction: torch.Tensor,
    target_alpha: float,
    semantic_beta: float,
    format_beta: float,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    layer_idx = int(key[1])
    layer = get_layers(model)[layer_idx]
    attn = layer.self_attn
    pre_rec = recipient_state.get("attn_o_inputs", {}).get(key)
    pre_donor = donor_state.get("attn_o_inputs", {}).get(key)
    if pre_rec is None or pre_donor is None or not hasattr(attn, "o_proj"):
        return []
    weight = attn.o_proj.weight.detach().float().cpu()
    n_heads = infer_num_heads(model, attn)
    if not n_heads:
        return []
    in_features = int(weight.shape[1])
    if in_features % int(n_heads) != 0:
        return []
    head_dim = in_features // int(n_heads)
    delta_pre = (pre_donor - pre_rec).float().cpu()
    target_w = unembed[int(target_id)].float().cpu()
    blocker_w = mean_unembed(unembed, format_ids, target_id)
    readout = target_w - blocker_w
    score_vals: list[float] = []
    raw_by_head: dict[int, torch.Tensor] = {}
    for head_id in range(int(n_heads)):
        start = head_id * head_dim
        end = start + head_dim
        raw_delta = torch.matmul(weight[:, start:end], delta_pre[start:end]).detach().cpu()
        raw_by_head[head_id] = raw_delta
        score_vals.append(float(torch.dot(raw_delta.float(), readout.float()).item()))
    selected = select_unit_ids(torch.tensor(score_vals, dtype=torch.float32), int(args.max_heads_per_component))
    rows = []
    for unit_rank, head_id in enumerate(selected, 1):
        raw_delta = raw_by_head[int(head_id)]
        unit_delta = exact_unit_format_delta(
            raw_delta,
            target_direction,
            semantic_direction,
            format_direction,
            identity_direction,
            target_alpha,
            semantic_beta,
            format_beta,
        )
        rows.append(
            {
                "unit_kind": "attention_head",
                "unit_id": int(head_id),
                "unit_rank": unit_rank,
                "raw_readout_score": score_vals[int(head_id)],
                "unit_format_delta_norm": float(unit_delta.float().norm().item()),
                "unit_delta": unit_delta,
                "num_heads": int(n_heads),
                "head_dim": int(head_dim),
            }
        )
    return rows


def mlp_unit_candidates(
    model,
    key: tuple[str, int],
    recipient_state: dict[str, Any],
    donor_state: dict[str, Any],
    unembed: torch.Tensor,
    target_id: int,
    format_ids: list[int],
    target_direction: torch.Tensor,
    semantic_direction: torch.Tensor,
    format_direction: torch.Tensor,
    identity_direction: torch.Tensor,
    target_alpha: float,
    semantic_beta: float,
    format_beta: float,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    layer_idx = int(key[1])
    layer = get_layers(model)[layer_idx]
    pre_rec = recipient_state.get("mlp_down_inputs", {}).get(key)
    pre_donor = donor_state.get("mlp_down_inputs", {}).get(key)
    if pre_rec is None or pre_donor is None or not hasattr(layer.mlp, "down_proj"):
        return []
    weight = layer.mlp.down_proj.weight.detach().float().cpu()
    delta_pre = (pre_donor - pre_rec).float().cpu()
    target_w = unembed[int(target_id)].float().cpu()
    blocker_w = mean_unembed(unembed, format_ids, target_id)
    readout = target_w - blocker_w
    coeff = torch.matmul(weight.T, readout.float())
    scores = delta_pre * coeff
    selected = select_unit_ids(scores, int(args.max_mlp_channels_per_component))
    rows = []
    for unit_rank, channel_id in enumerate(selected, 1):
        ch = int(channel_id)
        raw_delta = (weight[:, ch] * delta_pre[ch]).detach().cpu()
        unit_delta = exact_unit_format_delta(
            raw_delta,
            target_direction,
            semantic_direction,
            format_direction,
            identity_direction,
            target_alpha,
            semantic_beta,
            format_beta,
        )
        rows.append(
            {
                "unit_kind": "mlp_channel",
                "unit_id": ch,
                "unit_rank": unit_rank,
                "raw_readout_score": float(scores[ch].item()),
                "unit_format_delta_norm": float(unit_delta.float().norm().item()),
                "unit_delta": unit_delta,
                "mlp_intermediate_size": int(scores.numel()),
            }
        )
    return rows


def unit_candidates_for_component(
    model,
    key: tuple[str, int],
    recipient_state: dict[str, Any],
    donor_state: dict[str, Any],
    unembed: torch.Tensor,
    target_id: int,
    format_ids: list[int],
    target_direction: torch.Tensor,
    semantic_direction: torch.Tensor,
    format_direction: torch.Tensor,
    identity_direction: torch.Tensor,
    target_alpha: float,
    semantic_beta: float,
    format_beta: float,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    if key[0] == "attn":
        return attention_unit_candidates(
            model,
            key,
            recipient_state,
            donor_state,
            unembed,
            target_id,
            format_ids,
            target_direction,
            semantic_direction,
            format_direction,
            identity_direction,
            target_alpha,
            semantic_beta,
            format_beta,
            args,
        )
    if key[0] == "mlp":
        return mlp_unit_candidates(
            model,
            key,
            recipient_state,
            donor_state,
            unembed,
            target_id,
            format_ids,
            target_direction,
            semantic_direction,
            format_direction,
            identity_direction,
            target_alpha,
            semantic_beta,
            format_beta,
            args,
        )
    return []


def label_unit(row: dict[str, Any], args: argparse.Namespace) -> str:
    single_net = finite(row.get("single_transition_net_count_delta"))
    single_emerged = finite(row.get("single_transition_emerged_count"))
    single_resolved = finite(row.get("single_transition_resolved_count"))
    single_rate = finite(row.get("single_transition_emergence_rate_vs_base"))
    single_bias = finite(row.get("single_required_bias_delta_vs_semantic_base"))
    loo_net_loss = finite(row.get("loo_net_loss_vs_full"))
    if bool(row.get("single_token_closure_gain")):
        return "unit_token_closure"
    if (
        single_net < 0
        and single_resolved > single_emerged
        and single_rate <= float(args.max_emergence_rate_for_unit)
        and single_bias <= 0
        and loo_net_loss >= float(args.min_loo_net_loss)
    ):
        return "unit_closer_candidate_no_closure"
    if single_net < 0 and single_resolved > single_emerged and single_rate <= float(args.max_emergence_rate_for_unit):
        return "unit_weak_reducer"
    if single_emerged > single_resolved and single_net > 0:
        return "unit_new_blocker_or_deformer"
    return "unit_neutral_or_mixed"


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
        return [{"row_kind": "phase809_error", "model": args.model, "case_id": case["case_id"], "route_id": route["route_id"], "error": alpha0_error or "empty_alpha0_logits"}]
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
    semantic_snapshot_cache: dict[str, Any] | None = None
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
        semantic_logits, semantic_error = p801.run_logits_with_projected_route(model, device, recipient_ids, semantic_projected, recipient_answer_pos)
        if semantic_error or not semantic_logits.numel():
            rows.append({"row_kind": "phase809_error", "model": args.model, "case_id": case["case_id"], "route_id": route["route_id"], "error": semantic_error or "empty_semantic_logits"})
            continue
        semantic_snapshot_cache = p805.residual_snapshot(
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
        full_projected, _full_projection_metrics = p806.projected_state_multi(
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
            0.0,
        )
        full_logits, full_error = p801.run_logits_with_projected_route(model, device, recipient_ids, full_projected, recipient_answer_pos)
        if full_error or not full_logits.numel():
            rows.append({"row_kind": "phase809_error", "model": args.model, "case_id": case["case_id"], "route_id": route["route_id"], "error": full_error or "empty_full_logits"})
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
        full_bias_delta = finite(full_snapshot.get("residual_required_bias_to_clear_all")) - finite(
            semantic_snapshot_cache.get("residual_required_bias_to_clear_all")
        )
        for comp in route_components:
            key = (str(comp["component_kind"]), int(comp["layer"]))
            if key not in semantic_projected or key not in full_projected:
                continue
            units = unit_candidates_for_component(
                model,
                key,
                recipient_state,
                donor_state,
                unembed,
                target_id,
                direction_tokens["format_echo_ids"],
                target_direction,
                semantic_direction,
                format_direction,
                identity_direction,
                target_alpha,
                float(args.semantic_beta),
                float(args.format_beta),
                args,
            )
            for unit in units:
                unit_delta = unit.pop("unit_delta")
                single_projected = dict(semantic_projected)
                single_projected[key] = (semantic_projected[key].float() + unit_delta.float()).detach().cpu()
                loo_projected = dict(full_projected)
                loo_projected[key] = (full_projected[key].float() - unit_delta.float()).detach().cpu()
                single_logits, single_error = p801.run_logits_with_projected_route(model, device, recipient_ids, single_projected, recipient_answer_pos)
                loo_logits, loo_error = p801.run_logits_with_projected_route(model, device, recipient_ids, loo_projected, recipient_answer_pos)
                row: dict[str, Any] = {
                    "row_kind": "phase809_late_layer_head_channel_closer_decomposition",
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
                    "semantic_direction_mode": args.semantic_direction_mode,
                    "single_error": single_error,
                    "loo_error": loo_error,
                    "full_transition_net_count_delta": full_transition.get("transition_net_count_delta"),
                    "full_transition_resolved_count": full_transition.get("transition_resolved_count"),
                    "full_transition_emerged_count": full_transition.get("transition_emerged_count"),
                    "full_required_bias_delta_vs_semantic_base": full_bias_delta,
                    "full_token_closure_gain": bool(full_snapshot.get("token_closure_gain")),
                    "format_echo_direction_token_ids": direction_tokens["format_echo_ids"],
                    "direction_class_counts": direction_tokens["direction_class_counts"],
                    **{k: v for k, v in unit.items() if k != "unit_delta"},
                }
                if single_logits.numel():
                    row.update(
                        p808.transition_prefixed(
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
                    ) - finite(semantic_snapshot_cache.get("residual_required_bias_to_clear_all"))
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
                        p808.transition_prefixed(
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
                        semantic_snapshot_cache.get("residual_required_bias_to_clear_all")
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
                    row["loo_bias_loss_vs_full"] = loo_bias_delta - full_bias_delta
                row["phase809_label"] = label_unit(row, args)
                row["phase809_boundary"] = (
                    "This phase decomposes late-layer closer source components into attention heads and MLP channels. "
                    "It produces unit-level closure-solver input candidates, not a proof of token closure."
                )
                rows.append(row)
    return rows


def group_rows(rows: list[dict[str, Any]], fields: list[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("row_kind") == "phase809_late_layer_head_channel_closer_decomposition":
            groups[tuple(row.get(f) for f in fields)].append(row)
    out: list[dict[str, Any]] = []
    for key, vals in groups.items():
        payload = {field: value for field, value in zip(fields, key)}
        payload.update(
            {
                "n": len(vals),
                "case_n": len({v.get("case_id") for v in vals}),
                "mean_raw_readout_score": safe_mean([v.get("raw_readout_score") for v in vals]),
                "mean_unit_format_delta_norm": safe_mean([v.get("unit_format_delta_norm") for v in vals]),
                "mean_full_net_delta": safe_mean([v.get("full_transition_net_count_delta") for v in vals]),
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
                "label_counts": dict(Counter(v.get("phase809_label") for v in vals)),
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
        "phase": 809,
        "title": "Late-Layer Closer Source Decomposition into Head/Channel Units",
        "model": args.model,
        "round": args.round_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "attn_implementation": attn_impl,
        "dtype": "bfloat16",
        "quantization": "off",
        "n_rows": len(rows),
        "n_cases": len({r.get("case_id") for r in rows if r.get("row_kind") == "phase809_late_layer_head_channel_closer_decomposition"}),
        "n_routes": len(routes),
        "target_alpha_grid": parse_float_grid(args.target_alpha_grid, [0.75]),
        "semantic_beta": float(args.semantic_beta),
        "format_beta": float(args.format_beta),
        "max_heads_per_component": int(args.max_heads_per_component),
        "max_mlp_channels_per_component": int(args.max_mlp_channels_per_component),
        "by_unit": group_rows(rows, ["model", "unit_kind", "component_kind", "layer", "unit_id"])[:240],
        "by_component_unit_kind": group_rows(rows, ["model", "unit_kind", "component_kind", "layer"])[:160],
        "by_case_unit": group_rows(rows, ["model", "case_id", "unit_kind", "component_kind", "layer", "unit_id"])[:240],
        "strict_boundary": (
            "This phase produces unit-level candidates for a later closure solver. "
            "It is still not a minimal sufficient closure circuit."
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
        f"target_alpha={args.target_alpha_grid} semantic_beta={args.semantic_beta} format_beta={args.format_beta} "
        f"heads={args.max_heads_per_component} mlp_channels={args.max_mlp_channels_per_component}"
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
                log(f"{args.model}: head/channel closer decomposition {ci}/{len(selected)} cases; rows={len(rows)}")
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
    write_jsonl(out_dir / f"phase809_{args.model}_rows.jsonl", rows)
    write_json(out_dir / f"phase809_{args.model}_summary.json", summary)
    print(
        json.dumps(
            {
                "model": args.model,
                "round": args.round_name,
                "attn": attn_impl,
                "n_cases": summary["n_cases"],
                "n_rows": summary["n_rows"],
                "by_unit": summary["by_unit"][:16],
                "by_component_unit_kind": summary["by_component_unit_kind"][:12],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return summary


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        f"# Phase 809 Late-Layer Head/Channel Closer Decomposition ({payload['round']})",
        "",
        f"- Status: `{payload['status']}`",
        "- Boundary: unit-level candidates for closure-solver input, not final token closure.",
        "",
        "## By Unit",
        "",
        "| model | unit | rows | cases | single net | single resolved | single emerged | emergence rate | single bias | fmt supp | loo net loss | loo bias loss | closure | labels |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for model_name in MODELS:
        data = payload.get("model_summaries", {}).get(model_name)
        if not data:
            continue
        for row in data.get("by_unit", [])[:120]:
            label = f"{row.get('unit_kind')}:{row.get('component_kind')}:L{row.get('layer')}:u{row.get('unit_id')}"
            lines.append(
                f"| {model_name} | `{label}` | {row.get('n')} | {row.get('case_n')} | "
                f"{fmt(row.get('mean_single_net_delta'))} | {fmt(row.get('mean_single_resolved'))} | "
                f"{fmt(row.get('mean_single_emerged'))} | {fmt(row.get('mean_single_emergence_rate'))} | "
                f"{fmt(row.get('mean_single_bias_delta'))} | {fmt(row.get('mean_single_format_suppression'))} | "
                f"{fmt(row.get('mean_loo_net_loss_vs_full'))} | {fmt(row.get('mean_loo_bias_loss_vs_full'))} | "
                f"{fmt(row.get('single_token_closure_gain_rate'))} | "
                f"`{json.dumps(row.get('label_counts') or {}, ensure_ascii=False)}` |"
            )
    lines += [
        "",
        "## By Component Unit Kind",
        "",
        "| model | group | rows | cases | single net | single resolved | single emerged | loo net loss | labels |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for model_name in MODELS:
        data = payload.get("model_summaries", {}).get(model_name)
        if not data:
            continue
        for row in data.get("by_component_unit_kind", [])[:80]:
            label = f"{row.get('unit_kind')}:{row.get('component_kind')}:L{row.get('layer')}"
            lines.append(
                f"| {model_name} | `{label}` | {row.get('n')} | {row.get('case_n')} | "
                f"{fmt(row.get('mean_single_net_delta'))} | {fmt(row.get('mean_single_resolved'))} | "
                f"{fmt(row.get('mean_single_emerged'))} | {fmt(row.get('mean_loo_net_loss_vs_full'))} | "
                f"`{json.dumps(row.get('label_counts') or {}, ensure_ascii=False)}` |"
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def summarize_round(round_name: str) -> dict[str, Any]:
    out_dir = RESULT_ROOT / round_name
    payload: dict[str, Any] = {"phase": 809, "round": round_name, "status": "missing", "model_summaries": {}, "models": []}
    for model_name in MODELS:
        path = out_dir / f"phase809_{model_name}_summary.json"
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        payload["model_summaries"][model_name] = data
        payload["models"].append(model_name)
    payload["status"] = "complete" if len(payload["models"]) == len(MODELS) else "partial"
    write_json(out_dir / "phase809_cross_model_summary.json", payload)
    write_markdown(out_dir / "phase809_cross_model_summary.md", payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = p808.build_parser()
    parser.add_argument("--max-heads-per-component", type=int, default=6)
    parser.add_argument("--max-mlp-channels-per-component", type=int, default=8)
    parser.add_argument("--max-emergence-rate-for-unit", type=float, default=0.15)
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
