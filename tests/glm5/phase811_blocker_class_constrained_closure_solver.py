#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import itertools
import json
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
import phase806_format_echo_identity_residual_suppressor_search as p806  # noqa: E402
import phase807_readout_geometry_new_blocker_emergence_audit as p807  # noqa: E402
import phase808_readout_closer_source_localization as p808  # noqa: E402
import phase809_late_layer_head_channel_closer_decomposition as p809  # noqa: E402
from model_utils import release_model  # noqa: E402
from phase722_functional_head_atlas_causal_ablation import write_json, write_jsonl  # noqa: E402
from phase735_source_restricted_writer_validation import MODELS  # noqa: E402
from phase771_matched_causal_intervention_reliability_test import case_map_for  # noqa: E402
from phase773_instruction_source_disentanglement import fmt  # noqa: E402
from phase780_surface_form_component_localization import lm_head_weight  # noqa: E402
from phase795_multi_component_causal_fiber_closure import selected_route_components  # noqa: E402


RESULT_ROOT = Path("tests/result/phase811_blocker_class_constrained_closure_solver")
DEFAULT_CLASS_WEIGHTS = {
    "candidate_list_or_case_value": 1.8,
    "designated_contrast": 1.6,
    "semantic_or_lexical_competitor": 1.35,
    "echo_token": 1.25,
    "high_frequency_or_format": 1.15,
    "whitespace_or_newline": 1.05,
    "punctuation": 0.95,
    "number_or_symbol": 0.9,
    "special_token": 0.8,
    "other_token": 1.0,
}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def finite(value: Any, default: float = 0.0) -> float:
    val = p809.safe_float(value)
    return default if val is None else val


def parse_float_grid(text: str, fallback: list[float]) -> list[float]:
    return p809.parse_float_grid(text, fallback)


def item_label(item: dict[str, Any]) -> str:
    if item["item_kind"] == "identity_anchor":
        return f"identity_anchor:beta{item.get('identity_beta')}"
    return f"{item['unit_kind']}:{item['component_kind']}:L{item['layer']}:u{item['unit_id']}"


def logit_margin(logits: torch.Tensor, target_id: int) -> float:
    vals = logits.detach().float().cpu()
    target = float(vals[int(target_id)].item())
    mask = torch.ones_like(vals, dtype=torch.bool)
    mask[int(target_id)] = False
    other_max = float(vals[mask].max().item())
    return target - other_max


def class_weights(args: argparse.Namespace) -> dict[str, float]:
    weights = dict(DEFAULT_CLASS_WEIGHTS)
    overrides = str(getattr(args, "class_weight_overrides", "") or "").strip()
    if overrides:
        for chunk in overrides.split(","):
            if not chunk.strip() or ":" not in chunk:
                continue
            key, value = chunk.split(":", 1)
            try:
                weights[key.strip()] = float(value)
            except ValueError:
                continue
    return weights


def weighted_count(counts: dict[str, Any] | None, weights: dict[str, float]) -> float:
    total = 0.0
    for cls, count in (counts or {}).items():
        total += float(weights.get(str(cls), 1.0)) * finite(count)
    return total


def reconstruct_base_class_counts(row: dict[str, Any]) -> dict[str, int]:
    after = Counter({str(k): int(v) for k, v in (row.get("after_class_counts") or {}).items()})
    resolved = Counter({str(k): int(v) for k, v in (row.get("transition_resolved_class_counts") or {}).items()})
    emerged = Counter({str(k): int(v) for k, v in (row.get("transition_emerged_class_counts") or {}).items()})
    base: Counter[str] = Counter()
    for cls in set(after) | set(resolved) | set(emerged):
        base[cls] = max(int(after.get(cls, 0)) - int(emerged.get(cls, 0)) + int(resolved.get(cls, 0)), 0)
    return dict(base)


def class_metrics(row: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    weights = class_weights(args)
    base_counts = row.get("semantic_class_counts") or reconstruct_base_class_counts(row)
    after_counts = row.get("after_class_counts") or {}
    resolved_counts = row.get("transition_resolved_class_counts") or {}
    emerged_counts = row.get("transition_emerged_class_counts") or {}
    active_classes = sorted(str(c) for c, n in base_counts.items() if finite(n) > 0)
    reduced_classes = []
    unreduced_classes = []
    class_deltas: dict[str, int] = {}
    for cls in active_classes:
        before = int(finite(base_counts.get(cls)))
        after = int(finite(after_counts.get(cls)))
        delta = after - before
        class_deltas[cls] = delta
        if after < before:
            reduced_classes.append(cls)
        else:
            unreduced_classes.append(cls)
    weighted_before = weighted_count(base_counts, weights)
    weighted_after = weighted_count(after_counts, weights)
    weighted_resolved = weighted_count(resolved_counts, weights)
    weighted_emerged = weighted_count(emerged_counts, weights)
    coverage = len(reduced_classes) / max(len(active_classes), 1)
    return {
        "class_weights": weights,
        "class_weighted_before": weighted_before,
        "class_weighted_after": weighted_after,
        "class_weighted_delta": weighted_after - weighted_before,
        "class_weighted_resolved": weighted_resolved,
        "class_weighted_emerged": weighted_emerged,
        "class_active_count": len(active_classes),
        "class_reduced_count": len(reduced_classes),
        "class_unreduced_count": len(unreduced_classes),
        "class_reduction_coverage": coverage,
        "class_reduced_classes": reduced_classes,
        "class_unreduced_classes": unreduced_classes,
        "class_delta_counts": class_deltas,
    }


def full_snapshot(
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
    snap = p798.full_vocab_snapshot(
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
    snap["target_rank"] = int(snap.get("full_above_count") or 0) + 1
    snap["target_margin_vs_top_other"] = logit_margin(logits, target_id)
    snap["token_closure"] = int(snap.get("full_above_count") or 0) == 0
    return snap


def apply_items(
    base_projected: dict[tuple[str, int], torch.Tensor],
    items: list[dict[str, Any]],
    scale: float,
) -> dict[tuple[str, int], torch.Tensor]:
    projected = {key: val.detach().float().cpu() for key, val in base_projected.items()}
    for item in items:
        for key, delta in item["delta_by_key"].items():
            if key not in projected:
                continue
            projected[key] = (projected[key].float() + float(scale) * delta.float()).detach().cpu()
    return projected


def evaluate_projection(
    model,
    device,
    args: argparse.Namespace,
    semantic_logits: torch.Tensor,
    projected: dict[tuple[str, int], torch.Tensor],
    recipient_ids: list[int],
    answer_pos: int,
    target_id: int,
    contrast_id: int,
    prompt_text: str,
    candidate_ids: set[int],
    case_values: set[str],
    target_answer: str,
) -> tuple[torch.Tensor, dict[str, Any], dict[str, Any], str | None]:
    logits, error = p801.run_logits_with_projected_route(model, device, recipient_ids, projected, answer_pos)
    if error or not logits.numel():
        return logits, {}, {}, error or "empty_logits"
    transition = p807.transition_metrics(
        args,
        semantic_logits,
        logits,
        target_id,
        contrast_id,
        recipient_ids,
        prompt_text,
        candidate_ids,
        case_values,
    )
    snapshot = full_snapshot(
        args,
        logits,
        target_id,
        contrast_id,
        recipient_ids,
        prompt_text,
        candidate_ids,
        case_values,
        target_answer,
    )
    return logits, transition, snapshot, None


def objective(row: dict[str, Any], args: argparse.Namespace) -> float:
    metrics = row.get("class_metrics") or class_metrics(row, args)
    return (
        finite(metrics.get("class_weighted_after"), 1e9)
        + float(args.objective_lambda_l0) * finite(row.get("combo_size"), 0.0)
        + float(args.objective_mu_new) * finite(metrics.get("class_weighted_emerged"), 0.0)
        + float(args.objective_unreduced_class_penalty) * finite(metrics.get("class_unreduced_count"), 0.0)
        - float(args.objective_class_resolution_bonus) * finite(metrics.get("class_weighted_resolved"), 0.0)
        - float(args.objective_eta_margin) * finite(row.get("target_margin_vs_top_other"), 0.0)
    )


def label_solution(row: dict[str, Any], args: argparse.Namespace) -> str:
    if bool(row.get("token_closure")):
        return "class_constrained_token_closure"
    after_count = finite(row.get("after_full_above_count"), 999999.0)
    if after_count <= float(args.max_near_closure_blockers):
        return "class_constrained_near_closure_no_token"
    metrics = row.get("class_metrics") or class_metrics(row, args)
    coverage = finite(metrics.get("class_reduction_coverage"))
    resolved = finite(row.get("transition_resolved_count"))
    emerged = finite(row.get("transition_emerged_count"))
    net_delta = finite(row.get("transition_net_count_delta"))
    class_delta = finite(metrics.get("class_weighted_delta"))
    if (
        class_delta < 0
        and net_delta < 0
        and resolved > emerged
        and coverage >= float(args.min_class_coverage_rate)
    ):
        return "class_balanced_reducer_no_closure"
    if class_delta < 0 and net_delta < 0 and resolved > emerged:
        return "single_or_partial_class_reducer_no_closure"
    if net_delta > 0 and emerged > resolved:
        return "class_new_blocker_or_deformer"
    return "class_mixed_or_neutral"


def unit_items_for_route(
    model,
    route_components: list[dict[str, Any]],
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
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for comp in route_components:
        key = (str(comp["component_kind"]), int(comp["layer"]))
        units = p809.unit_candidates_for_component(
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
            float(args.semantic_beta),
            float(args.format_beta),
            args,
        )
        for unit in units:
            delta = unit.pop("unit_delta")
            payload = {
                "item_kind": "unit",
                "component_kind": key[0],
                "layer": int(key[1]),
                "delta_by_key": {key: delta.detach().float().cpu()},
                **unit,
            }
            payload["item_id"] = item_label(payload)
            out.append(payload)
    return out


def identity_anchor_items(
    semantic_projected: dict[tuple[str, int], torch.Tensor],
    full_projected: dict[tuple[str, int], torch.Tensor],
    identity_projected_by_beta: dict[float, dict[tuple[str, int], torch.Tensor]],
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for beta, identity_projected in identity_projected_by_beta.items():
        if beta == 0:
            continue
        delta_by_key: dict[tuple[str, int], torch.Tensor] = {}
        total_norm = 0.0
        for key, full_val in full_projected.items():
            if key not in identity_projected or key not in semantic_projected:
                continue
            delta = (identity_projected[key].float() - full_val.float()).detach().cpu()
            norm = float(delta.norm().item())
            if norm > 1e-8:
                delta_by_key[key] = delta
                total_norm += norm
        if delta_by_key:
            item = {
                "item_kind": "identity_anchor",
                "item_id": f"identity_anchor:beta{beta:g}",
                "identity_beta": float(beta),
                "unit_kind": "identity_anchor",
                "component_kind": "multi",
                "layer": -1,
                "unit_id": -1,
                "raw_readout_score": None,
                "unit_format_delta_norm": total_norm,
                "delta_by_key": delta_by_key,
            }
            items.append(item)
    return items


def row_from_eval(
    args: argparse.Namespace,
    case: dict[str, Any],
    route: dict[str, Any],
    recipient_variant: str,
    donor_variant: str,
    route_components: list[dict[str, Any]],
    semantic_snapshot: dict[str, Any],
    full_snapshot_row: dict[str, Any],
    full_transition: dict[str, Any],
    target_alpha: float,
    scale: float,
    items: list[dict[str, Any]],
    transition: dict[str, Any],
    snapshot: dict[str, Any],
    error: str | None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "row_kind": "phase811_blocker_class_constrained_closure_solver",
        "model": args.model,
        "round": args.round_name,
        "case_id": case["case_id"],
        "domain": case.get("domain"),
        "relation": case.get("relation"),
        "object": case.get("object"),
        "target_answer": case.get("answer"),
        "contrast_answer": case.get("contrast_answer"),
        "route_id": route["route_id"],
        "recipient_variant": recipient_variant,
        "compare_variant": donor_variant,
        "route_component_signature": p801.route_signature(route_components),
        "target_direction_alpha": float(target_alpha),
        "semantic_suppression_beta": float(args.semantic_beta),
        "format_echo_suppression_beta": float(args.format_beta),
        "combo_scale": float(scale),
        "combo_size": len(items),
        "combo_item_ids": [item["item_id"] for item in items],
        "combo_has_identity_anchor": any(item["item_kind"] == "identity_anchor" for item in items),
        "combo_identity_betas": [item.get("identity_beta") for item in items if item["item_kind"] == "identity_anchor"],
        "combo_error": error,
        "semantic_full_above_count": semantic_snapshot.get("full_above_count"),
        "semantic_required_bias_to_clear_all": semantic_snapshot.get("full_required_bias_to_clear_all"),
        "semantic_target_margin_vs_top_other": semantic_snapshot.get("target_margin_vs_top_other"),
        "semantic_class_counts": semantic_snapshot.get("full_above_class_counts"),
        "semantic_class_entropy": semantic_snapshot.get("full_above_class_entropy"),
        "full_closer_full_above_count": full_snapshot_row.get("full_above_count"),
        "full_closer_required_bias_to_clear_all": full_snapshot_row.get("full_required_bias_to_clear_all"),
        "full_closer_target_margin_vs_top_other": full_snapshot_row.get("target_margin_vs_top_other"),
        "full_closer_class_counts": full_snapshot_row.get("full_above_class_counts"),
        "full_closer_class_entropy": full_snapshot_row.get("full_above_class_entropy"),
        "full_closer_transition_net_count_delta": full_transition.get("transition_net_count_delta"),
        "full_closer_transition_resolved_count": full_transition.get("transition_resolved_count"),
        "full_closer_transition_emerged_count": full_transition.get("transition_emerged_count"),
    }
    if not error:
        row.update(
            {
                "after_full_above_count": snapshot.get("full_above_count"),
                "after_target_rank": snapshot.get("target_rank"),
                "after_required_bias_to_clear_all": snapshot.get("full_required_bias_to_clear_all"),
                "target_margin_vs_top_other": snapshot.get("target_margin_vs_top_other"),
                "token_closure": bool(snapshot.get("token_closure")),
                "after_class_counts": snapshot.get("full_above_class_counts"),
                "after_class_entropy": snapshot.get("full_above_class_entropy"),
                "after_rank_window": snapshot.get("full_rank_window", [])[: int(args.residual_rank_window_saved)],
                "transition_net_count_delta": transition.get("transition_net_count_delta"),
                "transition_resolved_count": transition.get("transition_resolved_count"),
                "transition_emerged_count": transition.get("transition_emerged_count"),
                "transition_emergence_rate_vs_base": transition.get("transition_emergence_rate_vs_base"),
                "transition_resolved_class_counts": transition.get("transition_resolved_class_counts"),
                "transition_emerged_class_counts": transition.get("transition_emerged_class_counts"),
            }
        )
        row["class_metrics"] = class_metrics(row, args)
        row["objective_score"] = objective(row, args)
        row["phase811_label"] = label_solution(row, args)
    else:
        row["objective_score"] = None
        row["phase811_label"] = "combo_error"
    row["phase811_boundary"] = (
        "This phase upgrades Phase 810 from total blocker reduction to blocker-class constrained solving. "
        "A positive result requires token closure or broad class-balanced reduction with controlled new blockers."
    )
    return row


def select_combo_pool(single_rows: list[dict[str, Any]], items: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    row_by_id = {tuple(row.get("combo_item_ids", [])): row for row in single_rows if not row.get("combo_error")}
    scored: list[tuple[float, dict[str, Any]]] = []
    per_class_best: dict[str, tuple[float, dict[str, Any]]] = {}
    for item in items:
        row = row_by_id.get((item["item_id"],))
        if not row:
            continue
        score = objective(row, args)
        # Prefer reducers even if they are not near closure.
        if finite(row.get("transition_net_count_delta")) < 0:
            score -= 5.0
        metrics = row.get("class_metrics") or class_metrics(row, args)
        for cls in metrics.get("class_reduced_classes") or []:
            class_delta = finite((metrics.get("class_delta_counts") or {}).get(cls))
            class_score = score + class_delta * float((metrics.get("class_weights") or {}).get(cls, 1.0))
            if cls not in per_class_best or class_score < per_class_best[cls][0]:
                per_class_best[cls] = (class_score, item)
        if bool(row.get("token_closure")):
            score -= 1000.0
        if item["item_kind"] == "identity_anchor":
            score -= 2.0
        scored.append((score, item))
    scored.sort(key=lambda x: x[0])
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for _score, item in sorted(per_class_best.values(), key=lambda x: x[0]):
        if item["item_id"] not in seen:
            selected.append(item)
            seen.add(item["item_id"])
    for _score, item in scored:
        if item["item_id"] not in seen:
            selected.append(item)
            seen.add(item["item_id"])
        if len(selected) >= int(args.max_combo_candidates):
            break
    for item in items:
        if item["item_kind"] == "identity_anchor" and item["item_id"] not in seen:
            selected.append(item)
            seen.add(item["item_id"])
    return selected[: max(int(args.max_combo_candidates), 1) + 4]


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
    answer_pos = len(recipient_ids) - 1
    recipient_groups = p796.source_groups_for_prompt(tokenizer, recipient_prompt, case, recipient_ids)
    recipient_candidate_ids = p796.candidate_position_ids(tokenizer, recipient_ids, recipient_groups)
    case_vals = p796.value_strings(case)
    recipient_state = p796.capture_answer_outputs_and_sources(model, tokenizer, device, recipient_prompt, component_keys)
    donor_state = p796.capture_answer_outputs_and_sources(model, tokenizer, device, donor_prompt, component_keys)
    route_components = selected_route_components(
        route,
        set(p796.parse_csv(args.route_component_kinds) or ["attn", "mlp"]),
        args.max_route_components,
    )
    target_direction = unembed[int(target_id)].float().cpu()
    alpha0_projected, _ = p802.projected_component_state_alpha(
        recipient_state,
        donor_state,
        route_components,
        target_direction,
        0.0,
    )
    alpha0_logits, alpha0_error = p801.run_logits_with_projected_route(model, device, recipient_ids, alpha0_projected, answer_pos)
    if alpha0_error or not alpha0_logits.numel():
        return [
            {
                "row_kind": "phase811_error",
                "model": args.model,
                "case_id": case["case_id"],
                "route_id": route["route_id"],
                "error": alpha0_error or "empty_alpha0_logits",
            }
        ]
    semantic_ids = p803.semantic_new_blocker_ids(
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
    semantic_direction = p804.semantic_direction_from_blockers(unembed, target_id, semantic_ids, args.semantic_direction_mode)
    rows: list[dict[str, Any]] = []
    for target_alpha in parse_float_grid(args.target_alpha_grid, [0.75]):
        semantic_projected, _ = p804.projected_state_target_semantic(
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
            answer_pos,
        )
        if semantic_error or not semantic_logits.numel():
            rows.append(
                {
                    "row_kind": "phase811_error",
                    "model": args.model,
                    "case_id": case["case_id"],
                    "route_id": route["route_id"],
                    "error": semantic_error or "empty_semantic_logits",
                }
            )
            continue
        semantic_snapshot = full_snapshot(
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
        full_projected, _ = p806.projected_state_multi(
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
        full_logits, full_error = p801.run_logits_with_projected_route(model, device, recipient_ids, full_projected, answer_pos)
        if full_error or not full_logits.numel():
            rows.append(
                {
                    "row_kind": "phase811_error",
                    "model": args.model,
                    "case_id": case["case_id"],
                    "route_id": route["route_id"],
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
        full_snapshot_row = full_snapshot(
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
        unit_items = unit_items_for_route(
            model,
            route_components,
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
            args,
        )
        identity_projected_by_beta: dict[float, dict[tuple[str, int], torch.Tensor]] = {}
        if args.include_identity_anchor:
            for beta in parse_float_grid(args.identity_anchor_beta_grid, [0.5, 1.0]):
                identity_projected, _ = p806.projected_state_multi(
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
                    float(beta),
                )
                identity_projected_by_beta[float(beta)] = identity_projected
        items = unit_items + identity_anchor_items(semantic_projected, full_projected, identity_projected_by_beta)
        if not items:
            continue

        single_rows: list[dict[str, Any]] = []
        for item in items:
            projected = apply_items(semantic_projected, [item], 1.0)
            _logits, transition, snapshot, error = evaluate_projection(
                model,
                device,
                args,
                semantic_logits,
                projected,
                recipient_ids,
                answer_pos,
                target_id,
                contrast_id,
                recipient_prompt,
                recipient_candidate_ids,
                case_vals,
                str(case.get("answer", "")),
            )
            row = row_from_eval(
                args,
                case,
                route,
                recipient_variant,
                donor_variant,
                route_components,
                semantic_snapshot,
                full_snapshot_row,
                full_transition,
                target_alpha,
                1.0,
                [item],
                transition,
                snapshot,
                error,
            )
            row["search_stage"] = "single_prefilter"
            single_rows.append(row)
            rows.append(row)

        pool = select_combo_pool(single_rows, items, args)
        combo_specs: list[tuple[tuple[dict[str, Any], ...], float]] = []
        for combo_size in range(2, int(args.max_combo_size) + 1):
            for combo in itertools.combinations(pool, combo_size):
                combo_specs.extend((combo, scale) for scale in parse_float_grid(args.combo_scale_grid, [1.0]))
        combo_specs.sort(
            key=lambda cs: sum(
                objective(next((r for r in single_rows if r.get("combo_item_ids") == [item["item_id"]]), {}), args)
                for item in cs[0]
            )
        )
        combo_specs = combo_specs[: int(args.max_combos_per_case_route)]

        for combo, scale in combo_specs:
            projected = apply_items(semantic_projected, list(combo), scale)
            _logits, transition, snapshot, error = evaluate_projection(
                model,
                device,
                args,
                semantic_logits,
                projected,
                recipient_ids,
                answer_pos,
                target_id,
                contrast_id,
                recipient_prompt,
                recipient_candidate_ids,
                case_vals,
                str(case.get("answer", "")),
            )
            row = row_from_eval(
                args,
                case,
                route,
                recipient_variant,
                donor_variant,
                route_components,
                semantic_snapshot,
                full_snapshot_row,
                full_transition,
                target_alpha,
                scale,
                list(combo),
                transition,
                snapshot,
                error,
            )
            row["search_stage"] = "combo_search"
            rows.append(row)
    return rows


def group_rows(rows: list[dict[str, Any]], fields: list[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("row_kind") == "phase811_blocker_class_constrained_closure_solver" and not row.get("combo_error"):
            groups[tuple(row.get(f) for f in fields)].append(row)
    out: list[dict[str, Any]] = []
    for key, vals in groups.items():
        payload = {field: value for field, value in zip(fields, key)}
        labels = Counter(v.get("phase811_label") for v in vals)
        payload.update(
            {
                "n": len(vals),
                "case_n": len({v.get("case_id") for v in vals}),
                "mean_combo_size": p809.safe_mean([v.get("combo_size") for v in vals]),
                "mean_after_full_above_count": p809.safe_mean([v.get("after_full_above_count") for v in vals]),
                "mean_after_required_bias": p809.safe_mean([v.get("after_required_bias_to_clear_all") for v in vals]),
                "mean_target_margin": p809.safe_mean([v.get("target_margin_vs_top_other") for v in vals]),
                "mean_transition_net_delta": p809.safe_mean([v.get("transition_net_count_delta") for v in vals]),
                "mean_transition_resolved": p809.safe_mean([v.get("transition_resolved_count") for v in vals]),
                "mean_transition_emerged": p809.safe_mean([v.get("transition_emerged_count") for v in vals]),
                "mean_objective_score": p809.safe_mean([v.get("objective_score") for v in vals]),
                "mean_class_weighted_after": p809.safe_mean([(v.get("class_metrics") or {}).get("class_weighted_after") for v in vals]),
                "mean_class_weighted_delta": p809.safe_mean([(v.get("class_metrics") or {}).get("class_weighted_delta") for v in vals]),
                "mean_class_coverage": p809.safe_mean([(v.get("class_metrics") or {}).get("class_reduction_coverage") for v in vals]),
                "mean_unreduced_classes": p809.safe_mean([(v.get("class_metrics") or {}).get("class_unreduced_count") for v in vals]),
                "token_closure_rate": p809.safe_rate([v.get("token_closure") for v in vals]),
                "identity_anchor_rate": p809.safe_rate([v.get("combo_has_identity_anchor") for v in vals]),
                "label_counts": dict(labels),
            }
        )
        out.append(payload)
    out.sort(
        key=lambda r: (
            r.get("token_closure_rate") or 0.0,
            -(r.get("mean_after_full_above_count") or 999999.0),
            -(r.get("mean_objective_score") or 999999.0),
        ),
        reverse=True,
    )
    return out


def summarize(rows: list[dict[str, Any]], args: argparse.Namespace, attn_impl: str, routes: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [r for r in rows if r.get("row_kind") == "phase811_blocker_class_constrained_closure_solver" and not r.get("combo_error")]
    best = sorted(valid, key=lambda r: (not bool(r.get("token_closure")), finite(r.get("objective_score"), 1e9)))[:80]
    return {
        "phase": 811,
        "title": "Blocker-Class Constrained Closure Solver",
        "model": args.model,
        "round": args.round_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "attn_implementation": attn_impl,
        "dtype": "bfloat16",
        "quantization": "off",
        "n_rows": len(rows),
        "n_valid_rows": len(valid),
        "n_cases": len({r.get("case_id") for r in valid}),
        "n_routes": len(routes),
        "token_closure_rows": sum(1 for r in valid if r.get("token_closure")),
        "best_rows": best,
        "by_label": dict(Counter(r.get("phase811_label") for r in valid)),
        "by_search_stage": group_rows(valid, ["model", "search_stage"])[:60],
        "by_combo_size": group_rows(valid, ["model", "search_stage", "combo_size"])[:80],
        "by_identity_anchor": group_rows(valid, ["model", "combo_has_identity_anchor"])[:80],
        "by_class_coverage": group_rows(valid, ["model", "search_stage", "combo_size", "combo_has_identity_anchor"])[:80],
        "strict_boundary": (
            "This phase tests whether the Phase 809/810 candidate space can reduce blocker classes broadly, "
            "not merely reduce the total blocker count. Token closure remains the strict success criterion."
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
        f"class_combo_k={args.max_combo_size} pool={args.max_combo_candidates} combos={args.max_combos_per_case_route}"
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
                log(f"{args.model}: class-constrained closure solver {ci}/{len(selected)} cases; rows={len(rows)}")
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
    write_jsonl(out_dir / f"phase811_{args.model}_rows.jsonl", rows)
    write_json(out_dir / f"phase811_{args.model}_summary.json", summary)
    print(
        json.dumps(
            {
                "model": args.model,
                "round": args.round_name,
                "attn": attn_impl,
                "n_cases": summary["n_cases"],
                "n_valid_rows": summary["n_valid_rows"],
                "token_closure_rows": summary["token_closure_rows"],
                "by_label": summary["by_label"],
                "best_rows": summary["best_rows"][:10],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return summary


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        f"# Phase 811 Blocker-Class Constrained Closure Solver ({payload['round']})",
        "",
        f"- Status: `{payload['status']}`",
        "- Boundary: class-constrained solver over Phase 809/810 candidate space; not global language closure.",
        "",
        "## Best Rows",
        "",
        "| model | stage | case | size | identity | above | class_after | class_delta | coverage | unreduced | net | emerged | closure | objective | label | items |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for model_name in MODELS:
        data = payload.get("model_summaries", {}).get(model_name)
        if not data:
            continue
        for row in data.get("best_rows", [])[:40]:
            items = " + ".join(row.get("combo_item_ids") or [])
            cm = row.get("class_metrics") or {}
            lines.append(
                f"| {model_name} | {row.get('search_stage')} | {row.get('case_id')} | {row.get('combo_size')} | "
                f"{int(bool(row.get('combo_has_identity_anchor')))} | {fmt(row.get('after_full_above_count'))} | "
                f"{fmt(cm.get('class_weighted_after'))} | {fmt(cm.get('class_weighted_delta'))} | "
                f"{fmt(cm.get('class_reduction_coverage'))} | {fmt(cm.get('class_unreduced_count'))} | "
                f"{fmt(row.get('transition_net_count_delta'))} | "
                f"{fmt(row.get('transition_emerged_count'))} | {int(bool(row.get('token_closure')))} | "
                f"{fmt(row.get('objective_score'))} | `{row.get('phase811_label')}` | `{items}` |"
            )
    lines += [
        "",
        "## By Label",
        "",
        "| model | labels | token closures | valid rows |",
        "|---|---|---:|---:|",
    ]
    for model_name in MODELS:
        data = payload.get("model_summaries", {}).get(model_name)
        if not data:
            continue
        lines.append(
            f"| {model_name} | `{json.dumps(data.get('by_label') or {}, ensure_ascii=False)}` | "
            f"{data.get('token_closure_rows')} | {data.get('n_valid_rows')} |"
        )
    lines += [
        "",
        "## By Combo Size And Coverage",
        "",
        "| model | stage | size | rows | cases | above | class_after | class_delta | coverage | unreduced | net | closure rate | labels |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for model_name in MODELS:
        data = payload.get("model_summaries", {}).get(model_name)
        if not data:
            continue
        for row in data.get("by_combo_size", [])[:40]:
            lines.append(
                f"| {model_name} | {row.get('search_stage')} | {row.get('combo_size')} | {row.get('n')} | "
                f"{row.get('case_n')} | {fmt(row.get('mean_after_full_above_count'))} | "
                f"{fmt(row.get('mean_class_weighted_after'))} | {fmt(row.get('mean_class_weighted_delta'))} | "
                f"{fmt(row.get('mean_class_coverage'))} | {fmt(row.get('mean_unreduced_classes'))} | "
                f"{fmt(row.get('mean_transition_net_delta'))} | "
                f"{fmt(row.get('token_closure_rate'))} | `{json.dumps(row.get('label_counts') or {}, ensure_ascii=False)}` |"
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def summarize_round(round_name: str) -> dict[str, Any]:
    out_dir = RESULT_ROOT / round_name
    payload: dict[str, Any] = {"phase": 811, "round": round_name, "status": "missing", "model_summaries": {}, "models": []}
    for model_name in MODELS:
        path = out_dir / f"phase811_{model_name}_summary.json"
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        payload["model_summaries"][model_name] = data
        payload["models"].append(model_name)
    payload["status"] = "complete" if len(payload["models"]) == len(MODELS) else "partial"
    write_json(out_dir / "phase811_cross_model_summary.json", payload)
    write_markdown(out_dir / "phase811_cross_model_summary.md", payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = p809.build_parser()
    parser.add_argument("--max-combo-size", type=int, default=3)
    parser.add_argument("--max-combo-candidates", type=int, default=10)
    parser.add_argument("--max-combos-per-case-route", type=int, default=80)
    parser.add_argument("--combo-scale-grid", default="1.0")
    parser.add_argument("--include-identity-anchor", action="store_true")
    parser.add_argument("--identity-anchor-beta-grid", default="0.5,1.0")
    parser.add_argument("--objective-lambda-l0", type=float, default=0.25)
    parser.add_argument("--objective-mu-new", type=float, default=0.2)
    parser.add_argument("--objective-eta-margin", type=float, default=0.5)
    parser.add_argument("--objective-unreduced-class-penalty", type=float, default=2.0)
    parser.add_argument("--objective-class-resolution-bonus", type=float, default=0.1)
    parser.add_argument("--min-class-coverage-rate", type=float, default=0.35)
    parser.add_argument("--class-weight-overrides", default="")
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
