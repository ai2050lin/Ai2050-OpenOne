#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

import phase846_geometry_boundary_equation_fitting as p846  # noqa: E402
import phase862_negative_blocker_sign_mechanism_audit as p862  # noqa: E402
import phase885_stable_boundary_minimality_cross_model_audit as p885  # noqa: E402
import phase901_stop_token_competitiveness_audit as p901  # noqa: E402
import phase903_protocol_continuation_field_mapping as p903  # noqa: E402
import phase909_l0_attention_source_span_eos_boundary_audit as p909  # noqa: E402
import phase910_prompt_preserving_termination_route_reconstruction as p910  # noqa: E402
import phase911_full_vocab_blocker_displacement_audit as p911  # noqa: E402
import phase912_finite_blocker_band_source_localization as p912  # noqa: E402
import phase913_route_preserving_blocker_band_disentanglement as p913  # noqa: E402


PHASE = 918
MODELS = ["qwen3", "glm4", "deepseek7b"]
PHASE915_ROOT = Path("tests/result/phase915_near_boundary_action_gate_search")
RESULT_ROOT = Path("tests/result/phase918_l39_mlp_channel_a_blocker_suppressor_localization")


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def parse_csv(raw: str) -> list[str]:
    return [part.strip() for part in str(raw).split(",") if part.strip()]


def parse_floats(raw: str) -> list[float]:
    return [float(part) for part in parse_csv(raw)]


def median(values: list[float | int | None]) -> float | None:
    cleaned = [float(value) for value in values if value is not None]
    return None if not cleaned else float(statistics.median(cleaned))


def mean(values: list[float | int | None]) -> float | None:
    cleaned = [float(value) for value in values if value is not None]
    return None if not cleaned else float(sum(cleaned) / len(cleaned))


def candidate_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row.get("case_id"),
        row.get("prompt_variant"),
        row.get("source_subset_key"),
        row.get("edit_mode"),
        row.get("eval_kind"),
        row.get("boundary_group_kind"),
        row.get("boundary_factor"),
    )


def score_phase915_row(row: dict[str, Any]) -> tuple[int, int, int, float, float]:
    margin = float(row.get("patched_eos_margin_vs_blocker") or -9999.0)
    margin_delta = float(row.get("eos_margin_delta_vs_boundary") or -9999.0)
    rank_gain = 0
    if row.get("eos_rank_delta_vs_boundary") is not None:
        rank_gain = max(0, -int(row["eos_rank_delta_vs_boundary"]))
    return (
        int(bool(row.get("weak_action_candidate"))),
        int(bool(row.get("rank_improved"))),
        rank_gain,
        margin,
        margin_delta,
    )


def select_phase915_candidates(model_name: str, args: argparse.Namespace) -> list[dict[str, Any]]:
    path = PHASE915_ROOT / args.phase915_round / f"phase915_{model_name}_rows.jsonl"
    rows = read_jsonl(path)
    selected = []
    for row in rows:
        if row.get("control_label") != args.source_control_label:
            continue
        if args.boundary_blocker_token and row.get("boundary_blocker_token") != args.boundary_blocker_token:
            continue
        if not (row.get("weak_action_candidate") or row.get("rank_improved")):
            continue
        selected.append(dict(row))
    buckets: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in selected:
        key = candidate_key(row)
        if key not in buckets or score_phase915_row(row) > score_phase915_row(buckets[key]):
            buckets[key] = row
    out = list(buckets.values())
    out.sort(
        key=lambda row: (
            score_phase915_row(row),
            str(row.get("eval_domain")),
            str(row.get("case_id")),
            str(row.get("prompt_variant")),
        ),
        reverse=True,
    )
    return out[: max(0, int(args.max_candidates_per_model))]


def capture_boundary_logits_and_mlp_activation(
    model,
    device: torch.device,
    current_ids: list[int],
    route_delta: torch.Tensor,
    boundary_spec: dict[str, Any],
    prompt_len: int,
    prefix_len: int,
    l4_mlp_groups: dict[str, list[int]],
    target_layer_idx: int,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    handles = p913.install_route_and_disentangle_hooks(
        model,
        route_delta,
        boundary_spec,
        prompt_len,
        prefix_len,
        len(current_ids),
        l4_mlp_groups,
    )
    down_proj = p913.mlp_down_proj(model, int(target_layer_idx))
    captured: dict[str, torch.Tensor] = {}

    def capture(_module, inputs):
        if inputs and torch.is_tensor(inputs[0]):
            tensor = inputs[0]
            if tensor.ndim >= 3:
                captured["activation"] = tensor[:, -1, :].detach().float().cpu()[0]
            elif tensor.ndim >= 2:
                captured["activation"] = tensor[-1, :].detach().float().cpu()
        return None

    if down_proj is not None:
        handles.append(down_proj.register_forward_pre_hook(capture))
    if not handles:
        return None, None
    try:
        logits = p903.logits_plain(model, device, current_ids)
    finally:
        for handle in handles:
            handle.remove()
    return logits, captured.get("activation")


def top_group_from_scores(candidate_idx: torch.Tensor, score: torch.Tensor, budget: int, largest: bool = True) -> list[int]:
    k = min(int(budget), int(score.numel()))
    if k <= 0:
        return []
    chosen_local = torch.topk(score, k=k, largest=largest).indices
    return [int(x) for x in candidate_idx.index_select(0, chosen_local).detach().cpu().tolist()]


def channel_groups_for_boundary_case(
    model,
    device: torch.device,
    layer_idx: int,
    activation: torch.Tensor | None,
    eos_id: int | None,
    blocker_id: int | None,
    band_ids: list[int],
    candidate_pool: int,
) -> tuple[dict[str, list[int]], dict[str, Any]]:
    if activation is None or eos_id is None or blocker_id is None:
        return {}, {}
    down_proj = p913.mlp_down_proj(model, int(layer_idx))
    if down_proj is None:
        return {}, {}
    act = activation.detach().float().cpu()
    if act.numel() == 0:
        return {}, {}
    pool_n = min(int(candidate_pool), int(act.numel()))
    abs_vals = torch.abs(act)
    top_abs = torch.topk(abs_vals, k=pool_n, largest=True).indices
    low_abs = torch.topk(abs_vals, k=pool_n, largest=False).indices
    groups: dict[str, list[int]] = {
        "top_abs_64": [int(x) for x in top_abs[: min(64, len(top_abs))].tolist()],
        "low_abs_64": [int(x) for x in low_abs[: min(64, len(low_abs))].tolist()],
    }
    valid_band = [int(x) for x in band_ids if int(x) not in {int(eos_id), int(blocker_id)}]
    token_ids = [int(eos_id), int(blocker_id)] + valid_band
    token_rows = p913.lm_head_rows(model, token_ids, device)
    diagnostics: dict[str, Any] = {
        "activation_abs_top": float(abs_vals[top_abs[0]].item()) if len(top_abs) else None,
        "activation_abs_median": float(torch.median(abs_vals).item()),
        "candidate_pool_used": int(pool_n),
    }
    if token_rows is None or token_rows.shape[0] < 2:
        return groups, diagnostics
    eos_row = token_rows[0:1]
    blocker_row = token_rows[1:2]
    band_rows = token_rows[2:]
    candidate_idx = top_abs.to(device=down_proj.weight.device)
    down_cols = down_proj.weight.index_select(1, candidate_idx).detach().to(device=device, dtype=torch.float32)
    act_sub = act.index_select(0, top_abs).to(device=device, dtype=torch.float32)
    eos_proj = torch.matmul(eos_row, down_cols).squeeze(0)
    blocker_proj = torch.matmul(blocker_row, down_cols).squeeze(0)
    band_proj = torch.matmul(band_rows, down_cols).mean(dim=0) if band_rows.numel() else blocker_proj
    eos_support = act_sub * eos_proj
    blocker_support = act_sub * (blocker_proj - eos_proj)
    blocker_logit_support = act_sub * blocker_proj
    margin_support = act_sub * (eos_proj - blocker_proj)
    band_blocker_support = act_sub * (band_proj - eos_proj)
    recipes = [
        ("eos_support_32", eos_support, 32, True),
        ("eos_support_64", eos_support, 64, True),
        ("a_blocker_support_32", blocker_support, 32, True),
        ("a_blocker_support_64", blocker_support, 64, True),
        ("a_logit_support_64", blocker_logit_support, 64, True),
        ("margin_support_pos_32", margin_support, 32, True),
        ("margin_support_pos_64", margin_support, 64, True),
        ("margin_support_neg_32", margin_support, 32, False),
        ("margin_support_neg_64", margin_support, 64, False),
        ("band_blocker_support_64", band_blocker_support, 64, True),
    ]
    for name, score, budget, largest in recipes:
        chosen = top_group_from_scores(candidate_idx, score, budget, largest=largest)
        if not chosen:
            continue
        groups[name] = chosen
        local = torch.tensor([int((candidate_idx == int(x)).nonzero()[0].item()) for x in chosen], dtype=torch.long, device=score.device)
        diagnostics[f"{name}_mean_score"] = float(score.index_select(0, local).mean().item())
        diagnostics[f"{name}_max_score"] = float(score.index_select(0, local).max().item())
        diagnostics[f"{name}_min_score"] = float(score.index_select(0, local).min().item())
    return groups, diagnostics


def channel_specs(args: argparse.Namespace) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = [
        {
            "control_label": "boundary_precondition_only",
            "control_family": "boundary_precondition",
            "control_kind": "boundary_only",
            "neural_intervention": True,
        }
    ]
    up_groups = parse_csv(args.up_groups)
    down_groups = parse_csv(args.down_groups)
    general_groups = parse_csv(args.general_groups)
    for group_kind in up_groups:
        for factor in parse_floats(args.up_factors):
            specs.append(
                {
                    "control_label": f"L{args.target_layer}_mlp_channels_{group_kind}_scale_{factor:g}",
                    "control_family": "l39_channel_amplify",
                    "control_kind": "target_mlp_channel_group_scale",
                    "layer_idx": int(args.target_layer),
                    "group_kind": group_kind,
                    "factor": float(factor),
                    "neural_intervention": True,
                }
            )
    for group_kind in down_groups:
        for factor in parse_floats(args.down_factors):
            specs.append(
                {
                    "control_label": f"L{args.target_layer}_mlp_channels_{group_kind}_scale_{factor:g}",
                    "control_family": "l39_channel_suppress",
                    "control_kind": "target_mlp_channel_group_scale",
                    "layer_idx": int(args.target_layer),
                    "group_kind": group_kind,
                    "factor": float(factor),
                    "neural_intervention": True,
                }
            )
    for group_kind in general_groups:
        for factor in parse_floats(args.general_factors):
            specs.append(
                {
                    "control_label": f"L{args.target_layer}_mlp_channels_{group_kind}_scale_{factor:g}",
                    "control_family": "l39_channel_control",
                    "control_kind": "target_mlp_channel_group_scale",
                    "layer_idx": int(args.target_layer),
                    "group_kind": group_kind,
                    "factor": float(factor),
                    "neural_intervention": True,
                }
            )
    return specs


def logits_with_boundary_and_channel(
    model,
    device: torch.device,
    current_ids: list[int],
    route_delta: torch.Tensor,
    boundary_spec: dict[str, Any],
    prompt_len: int,
    prefix_len: int,
    l4_mlp_groups: dict[str, list[int]],
    channel_groups: dict[str, list[int]],
    spec: dict[str, Any],
) -> torch.Tensor | None:
    handles = p913.install_route_and_disentangle_hooks(
        model,
        route_delta,
        boundary_spec,
        prompt_len,
        prefix_len,
        len(current_ids),
        l4_mlp_groups,
    )
    if spec.get("control_kind") == "target_mlp_channel_group_scale":
        group = channel_groups.get(str(spec.get("group_kind"))) or []
        if not group:
            for handle in handles:
                handle.remove()
            return None
        handles.extend(
            p913.install_mlp_channel_group_scale(
                model,
                int(spec.get("layer_idx") or 39),
                group,
                1.0 if spec.get("factor") is None else float(spec.get("factor")),
            )
        )
    if not handles:
        return None
    try:
        return p903.logits_plain(model, device, current_ids)
    finally:
        for handle in handles:
            handle.remove()


def row_from_logits(
    tokenizer,
    case: dict[str, Any],
    source_row: dict[str, Any],
    spec: dict[str, Any],
    prefix_ids: list[int],
    prefix_text: str,
    route_metrics: dict[str, Any],
    boundary_metrics: dict[str, Any],
    patched_metrics: dict[str, Any],
    boundary_logits: torch.Tensor,
    patched_logits: torch.Tensor,
    boundary_top_rows: list[dict[str, Any]],
    patched_top_rows: list[dict[str, Any]],
    boundary_blocker_ids: list[int],
    route_delta_norm: float,
    channel_groups: dict[str, list[int]],
    channel_diag: dict[str, Any],
) -> dict[str, Any]:
    boundary_blocker = p910.first_non_eos_top(boundary_top_rows)
    patched_blocker = p910.first_non_eos_top(patched_top_rows)
    boundary_rank = boundary_metrics.get("eos_rank")
    patched_rank = patched_metrics.get("eos_rank")
    boundary_eos_logit = boundary_metrics.get("eos_best_logit")
    patched_eos_logit = patched_metrics.get("eos_best_logit")
    boundary_margin = p911.eos_margin_vs_blocker(boundary_metrics, boundary_blocker)
    patched_margin = p911.eos_margin_vs_blocker(patched_metrics, patched_blocker)
    boundary_blocker_id = boundary_blocker.get("token_id") if boundary_blocker else None
    boundary_blocker_logit = p911.token_logit(boundary_logits, boundary_blocker_id)
    boundary_blocker_after = p911.token_logit(patched_logits, boundary_blocker_id)
    rank_delta = None if boundary_rank is None or patched_rank is None else int(patched_rank) - int(boundary_rank)
    margin_delta = None if boundary_margin is None or patched_margin is None else float(patched_margin - boundary_margin)
    eos_delta = None if boundary_eos_logit is None or patched_eos_logit is None else float(patched_eos_logit - boundary_eos_logit)
    band_before = p912.stats_for_ids(boundary_logits, boundary_blocker_ids[:16])
    band_after = p912.stats_for_ids(patched_logits, boundary_blocker_ids[:16])
    band16_delta = None if band_before["mean"] is None or band_after["mean"] is None else float(band_after["mean"] - band_before["mean"])
    group_kind = spec.get("group_kind")
    group_ids = channel_groups.get(str(group_kind)) if group_kind else None
    eos_top1 = bool(patched_rank == 1)
    blocker_delta = None if boundary_blocker_logit is None or boundary_blocker_after is None else float(boundary_blocker_after - boundary_blocker_logit)
    return {
        "phase": PHASE,
        "row_kind": "phase918_l39_mlp_channel_a_blocker_suppressor_row",
        "model": source_row.get("model"),
        "source_phase": 915,
        "source_control_label": source_row.get("control_label"),
        "source_key": source_row.get("source_key"),
        "source_subset_key": source_row.get("source_subset_key"),
        "edit_mode": source_row.get("edit_mode"),
        "eval_kind": source_row.get("eval_kind"),
        "eval_domain": source_row.get("eval_domain"),
        "case_id": source_row.get("case_id"),
        "object": source_row.get("object"),
        "canonical_answer": source_row.get("canonical_answer"),
        "prompt_variant": source_row.get("prompt_variant"),
        "prefix_text": prefix_text,
        "boundary_group_kind": source_row.get("boundary_group_kind"),
        "boundary_factor": source_row.get("boundary_factor"),
        "source_l39_margin_delta": source_row.get("eos_margin_delta_vs_boundary"),
        "source_l39_eos_delta": source_row.get("eos_logit_delta_vs_boundary"),
        "source_l39_blocker_delta": source_row.get("boundary_blocker_logit_delta"),
        "control_label": spec.get("control_label"),
        "control_family": spec.get("control_family"),
        "control_kind": spec.get("control_kind"),
        "layer_idx": spec.get("layer_idx"),
        "group_kind": group_kind,
        "factor": spec.get("factor"),
        "neural_intervention": bool(spec.get("neural_intervention")),
        "prompt_input_intact": True,
        "prompt_all_zero_used_as_test_control": False,
        "route_delta_norm": route_delta_norm,
        "route_eos_rank": route_metrics.get("eos_rank"),
        "boundary_eos_rank": boundary_rank,
        "boundary_eos_logit": boundary_eos_logit,
        "boundary_eos_margin_vs_blocker": boundary_margin,
        "boundary_eos_top5": bool(boundary_rank is not None and int(boundary_rank) <= 5),
        "boundary_blocker_id": boundary_blocker_id,
        "boundary_blocker_token": boundary_blocker.get("token") if boundary_blocker else None,
        "boundary_blocker_logit": boundary_blocker_logit,
        "patched_eos_rank": patched_rank,
        "patched_eos_logit": patched_eos_logit,
        "patched_eos_top1": eos_top1,
        "patched_eos_top5": bool(patched_rank is not None and int(patched_rank) <= 5),
        "patched_eos_top10": bool(patched_rank is not None and int(patched_rank) <= 10),
        "patched_eos_margin_vs_blocker": patched_margin,
        "patched_eos_margin_nonnegative": bool(patched_margin is not None and patched_margin >= 0),
        "patched_blocker_token": patched_blocker.get("token") if patched_blocker else None,
        "patched_blocker_logit": patched_blocker.get("logit") if patched_blocker else None,
        "eos_rank_delta_vs_boundary": rank_delta,
        "eos_logit_delta_vs_boundary": eos_delta,
        "eos_margin_delta_vs_boundary": margin_delta,
        "boundary_blocker_logit_after_patch": boundary_blocker_after,
        "boundary_blocker_logit_delta": blocker_delta,
        "boundary_band16_mean_delta": band16_delta,
        "boundary_blocker_suppressed": bool(blocker_delta is not None and blocker_delta < 0),
        "promoted_margin_from_negative": bool(
            boundary_margin is not None and boundary_margin < 0 and patched_margin is not None and patched_margin >= 0
        ),
        "promoted_top1_from_non_top1": bool(boundary_rank is not None and int(boundary_rank) > 1 and patched_rank == 1),
        "promoted_top5_from_non_top5": bool(
            boundary_rank is not None and int(boundary_rank) > 5 and patched_rank is not None and int(patched_rank) <= 5
        ),
        "rank_improved": bool(rank_delta is not None and rank_delta < 0),
        "weak_channel_candidate": bool(
            rank_delta is not None
            and rank_delta < 0
            and eos_delta is not None
            and eos_delta >= 0
            and margin_delta is not None
            and margin_delta > 0
        ),
        "strict_clean_candidate": p911.strict_clean_candidate(tokenizer, case, prefix_ids, eos_top1),
        "channel_group_size": len(group_ids or []),
        "channel_group_preview": [int(x) for x in (group_ids or [])[:16]],
        "channel_diag": channel_diag if group_kind else {},
        "boundary_top8": boundary_top_rows[:8],
        "patched_top8": patched_top_rows[:8],
    }


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    channel_rows = [row for row in rows if row.get("control_kind") == "target_mlp_channel_group_scale"]
    boundary_rows = [row for row in rows if row.get("control_kind") == "boundary_only"]
    return {
        "rows": len(rows),
        "boundary_rows": len(boundary_rows),
        "channel_rows": len(channel_rows),
        "boundary_top1": sum(1 for row in boundary_rows if row.get("patched_eos_top1")),
        "boundary_top5": sum(1 for row in boundary_rows if row.get("patched_eos_top5")),
        "boundary_margin_nonnegative": sum(1 for row in boundary_rows if row.get("patched_eos_margin_nonnegative")),
        "channel_top1": sum(1 for row in channel_rows if row.get("patched_eos_top1")),
        "channel_top5": sum(1 for row in channel_rows if row.get("patched_eos_top5")),
        "channel_top10": sum(1 for row in channel_rows if row.get("patched_eos_top10")),
        "channel_margin_nonnegative": sum(1 for row in channel_rows if row.get("patched_eos_margin_nonnegative")),
        "channel_promoted_margin": sum(1 for row in channel_rows if row.get("promoted_margin_from_negative")),
        "channel_promoted_top1": sum(1 for row in channel_rows if row.get("promoted_top1_from_non_top1")),
        "channel_promoted_top5": sum(1 for row in channel_rows if row.get("promoted_top5_from_non_top5")),
        "channel_rank_improved": sum(1 for row in channel_rows if row.get("rank_improved")),
        "weak_channel_candidate": sum(1 for row in channel_rows if row.get("weak_channel_candidate")),
        "channel_blocker_suppressed": sum(1 for row in channel_rows if row.get("boundary_blocker_suppressed")),
        "channel_strict_clean_candidate": sum(1 for row in channel_rows if row.get("strict_clean_candidate")),
        "median_channel_margin_delta": median([row.get("eos_margin_delta_vs_boundary") for row in channel_rows]),
        "mean_channel_margin_delta": mean([row.get("eos_margin_delta_vs_boundary") for row in channel_rows]),
        "mean_channel_eos_delta": mean([row.get("eos_logit_delta_vs_boundary") for row in channel_rows]),
        "median_channel_blocker_delta": median([row.get("boundary_blocker_logit_delta") for row in channel_rows]),
        "boundary_blocker_tokens_top12": dict(Counter(str(row.get("boundary_blocker_token")) for row in rows).most_common(12)),
        "patched_blocker_tokens_top12": dict(Counter(str(row.get("patched_blocker_token")) for row in channel_rows).most_common(12)),
    }


def summarize_by_control(rows: list[dict[str, Any]], limit: int = 120) -> list[dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[str(row.get("control_label"))].append(row)
    out = []
    for label, vals in buckets.items():
        summary = summarize_rows(vals)
        first = vals[0]
        summary.update(
            {
                "control_label": label,
                "control_family": first.get("control_family"),
                "control_kind": first.get("control_kind"),
                "layer_idx": first.get("layer_idx"),
                "group_kind": first.get("group_kind"),
                "factor": first.get("factor"),
            }
        )
        out.append(summary)
    out.sort(
        key=lambda row: (
            row.get("channel_strict_clean_candidate") or 0,
            row.get("channel_promoted_top1") or 0,
            row.get("channel_top1") or 0,
            row.get("channel_promoted_margin") or 0,
            row.get("channel_margin_nonnegative") or 0,
            row.get("channel_promoted_top5") or 0,
            row.get("weak_channel_candidate") or 0,
            row.get("channel_rank_improved") or 0,
            row.get("channel_blocker_suppressed") or 0,
            row.get("median_channel_margin_delta") or -9999,
        ),
        reverse=True,
    )
    return out[:limit]


def summarize_by_group(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("control_kind") != "target_mlp_channel_group_scale":
            continue
        buckets[(str(row.get("group_kind")), str(row.get("control_family")))].append(row)
    out = []
    for (group_kind, family), vals in buckets.items():
        summary = summarize_rows(vals)
        summary.update({"group_kind": group_kind, "control_family": family})
        out.append(summary)
    out.sort(
        key=lambda row: (
            row.get("channel_strict_clean_candidate") or 0,
            row.get("channel_top1") or 0,
            row.get("channel_margin_nonnegative") or 0,
            row.get("channel_promoted_top5") or 0,
            row.get("weak_channel_candidate") or 0,
            row.get("channel_rank_improved") or 0,
            row.get("channel_blocker_suppressed") or 0,
            row.get("median_channel_margin_delta") or -9999,
        ),
        reverse=True,
    )
    return out


def summarize_model(model_name: str, rows: list[dict[str, Any]], selected_count: int, specs_count: int, attn_impl: str | None) -> dict[str, Any]:
    overall = summarize_rows(rows)
    if selected_count == 0:
        evidence = "no_phase915_l39_candidates"
    elif overall["channel_strict_clean_candidate"] > 0:
        evidence = "l39_channel_strict_clean_candidate_found"
    elif overall["channel_promoted_top1"] > 0 or overall["channel_top1"] > 0:
        evidence = "l39_channel_top1_candidate_found"
    elif overall["channel_promoted_margin"] > 0 or overall["channel_margin_nonnegative"] > 0:
        evidence = "l39_channel_margin_candidate_found"
    elif overall["channel_promoted_top5"] > 0 or overall["weak_channel_candidate"] > 0:
        evidence = "l39_channel_partial_boundary_movement_only"
    elif overall["channel_blocker_suppressed"] > 0:
        evidence = "l39_channel_blocker_suppression_without_eos_closure"
    else:
        evidence = "no_l39_channel_candidate_found"
    return {
        "phase": PHASE,
        "title": "L39 MLP Channel-level a Blocker Suppressor Localization",
        "model": model_name,
        "status": "complete",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "attn_implementation": attn_impl,
        "selected_phase915_l39_candidates": selected_count,
        "channel_spec_count": specs_count,
        "overall": overall,
        "control_summaries": summarize_by_control(rows),
        "group_summaries": summarize_by_group(rows),
        "evidence_label": evidence,
        "boundary": (
            "Phase918 fixes the Phase915 route+L4 boundary state and decomposes the L39 MLP down_proj "
            "input into EOS-support, a-blocker-support, and margin-support channel groups. It tests channel "
            "scales rather than whole-component L39 output scaling."
        ),
    }


def eval_model(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = RESULT_ROOT / args.round_name
    out_dir.mkdir(parents=True, exist_ok=True)
    selected = select_phase915_candidates(args.model, args)
    if args.dry_run or not selected:
        empty_summary = summarize_model(args.model, [], len(selected), 0, None)
        payload = {
            "phase": PHASE,
            "model": args.model,
            "status": "dry_run" if selected else "no_phase915_l39_candidates",
            "selected_phase915_l39_candidates": len(selected),
            "preview": selected[:20],
            "overall": empty_summary["overall"],
            "control_summaries": [],
            "group_summaries": [],
            "evidence_label": empty_summary["evidence_label"],
        }
        p846.write_json(out_dir / f"phase918_{args.model}_summary.json", payload)
        p846.write_jsonl(out_dir / f"phase918_{args.model}_rows.jsonl", [])
        print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
        return payload
    case_map = {str(case.get("case_id")): case for case in p885.extended_cases()}
    model = None
    tokenizer = None
    rows: list[dict[str, Any]] = []
    attn_impl = None
    all_specs: list[dict[str, Any]] = []
    try:
        model, tokenizer, device, attn_impl = p862.p844.p828.p796.load_model_bf16_prefer_flash(
            args.model, args.attn_implementations
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        groups = p903.protocol_category_groups(tokenizer)
        all_specs = channel_specs(args)
        prompt_cache: dict[tuple[str, str], list[int]] = {}
        for idx, source_row in enumerate(selected, 1):
            case = case_map.get(str(source_row.get("case_id")))
            if not case:
                continue
            prompt_key = (str(source_row.get("case_id")), str(source_row.get("prompt_variant")))
            if prompt_key not in prompt_cache:
                prompt = p885.prompt_for_case(case, str(source_row.get("prompt_variant")))
                prompt_cache[prompt_key] = p862.p844.encode_prompt(tokenizer, prompt)
            prompt_ids = prompt_cache[prompt_key]
            gears = p903.parse_gears(str(source_row.get("source_subset_key")))
            _prefix_logits, prefix_ids, prefix_text, _answer_seen = p901.logits_after_answer_prefix(
                model,
                tokenizer,
                device,
                prompt_ids,
                gears,
                str(source_row.get("edit_mode")),
                case,
                int(args.max_prefix_tokens),
                float(args.scale_up_factor),
            )
            current_ids = [int(x) for x in prompt_ids] + [int(x) for x in prefix_ids]
            answer_logits = p903.logits_plain(model, device, current_ids)
            answer_metrics = p903.state_metrics(tokenizer, answer_logits, groups)
            period_id = answer_metrics.get("period_best_id") or ((groups.get("period") or [None])[0])
            if period_id is None:
                continue
            period_ids = current_ids + [int(period_id)]
            _baseline_logits, base_vec = p910.logits_and_l0_vector(model, device, period_ids)
            prompt_zero_handles = p909.install_attention_input_span_scale(model, 0, 0, len(prompt_ids), 0.0)
            _prompt_zero_logits, prompt_zero_vec = p910.logits_and_l0_vector(model, device, period_ids, prompt_zero_handles)
            if base_vec is None or prompt_zero_vec is None:
                continue
            route_delta = prompt_zero_vec - base_vec
            route_delta_norm = float(torch.linalg.vector_norm(route_delta).item())
            route_logits, l4_activation = p913.capture_route_logits_and_mlp_activation(model, device, period_ids, route_delta, 4)
            if route_logits is None:
                continue
            route_metrics = p903.state_metrics(tokenizer, route_logits, groups)
            route_top_rows = p910.topk_tokens(tokenizer, route_logits, groups, max(64, int(args.band_size)))
            route_band32_ids = p911.top_non_eos_ids(route_top_rows, int(args.band_size))
            route_band16_ids = route_band32_ids[: min(16, len(route_band32_ids))]
            l4_mlp_groups, _l4_mlp_diag = p913.mlp_channel_groups_for_case(
                model,
                device,
                l4_activation,
                route_metrics.get("eos_best_id"),
                route_band16_ids,
                route_band32_ids,
                int(args.l4_candidate_pool),
            )
            boundary_factor = source_row.get("boundary_factor")
            if boundary_factor is None:
                continue
            boundary_spec = {
                "control_label": f"L4_mlp_channels_top_abs_64_scale_{float(boundary_factor):g}",
                "control_kind": "mlp_channel_group_scale",
                "layer_idx": 4,
                "group_kind": "top_abs_64",
                "factor": float(boundary_factor),
            }
            boundary_logits, l39_activation = capture_boundary_logits_and_mlp_activation(
                model,
                device,
                period_ids,
                route_delta,
                boundary_spec,
                len(prompt_ids),
                len(prefix_ids),
                l4_mlp_groups,
                int(args.target_layer),
            )
            if boundary_logits is None:
                continue
            boundary_metrics = p903.state_metrics(tokenizer, boundary_logits, groups)
            boundary_top_rows = p910.topk_tokens(tokenizer, boundary_logits, groups, max(64, int(args.band_size)))
            boundary_blocker = p910.first_non_eos_top(boundary_top_rows)
            boundary_blocker_ids = p911.top_non_eos_ids(boundary_top_rows, int(args.band_size))
            channel_groups, channel_diag = channel_groups_for_boundary_case(
                model,
                device,
                int(args.target_layer),
                l39_activation,
                boundary_metrics.get("eos_best_id"),
                boundary_blocker.get("token_id") if boundary_blocker else None,
                boundary_blocker_ids,
                int(args.channel_candidate_pool),
            )
            for spec in all_specs:
                if spec.get("control_kind") == "boundary_only":
                    patched_logits = boundary_logits
                else:
                    if not channel_groups.get(str(spec.get("group_kind"))):
                        continue
                    patched_logits = logits_with_boundary_and_channel(
                        model,
                        device,
                        period_ids,
                        route_delta,
                        boundary_spec,
                        len(prompt_ids),
                        len(prefix_ids),
                        l4_mlp_groups,
                        channel_groups,
                        spec,
                    )
                if patched_logits is None:
                    continue
                patched_metrics = p903.state_metrics(tokenizer, patched_logits, groups)
                patched_top_rows = p910.topk_tokens(tokenizer, patched_logits, groups, 16)
                rows.append(
                    row_from_logits(
                        tokenizer,
                        case,
                        source_row,
                        spec,
                        prefix_ids,
                        prefix_text,
                        route_metrics,
                        boundary_metrics,
                        patched_metrics,
                        boundary_logits,
                        patched_logits,
                        boundary_top_rows,
                        patched_top_rows,
                        boundary_blocker_ids,
                        route_delta_norm,
                        channel_groups,
                        channel_diag,
                    )
                )
            if idx % max(1, int(args.log_every)) == 0 or idx == len(selected):
                log(f"{args.model}/{args.round_name}: candidate={idx}/{len(selected)} rows={len(rows)}")
    finally:
        if model is not None:
            p862.p844.p828.release_model(model)
        if tokenizer is not None:
            del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    payload = summarize_model(args.model, rows, len(selected), len(all_specs), attn_impl)
    p846.write_json(out_dir / f"phase918_{args.model}_summary.json", payload)
    p846.write_jsonl(out_dir / f"phase918_{args.model}_rows.jsonl", rows)
    print(json.dumps({"phase": PHASE, "model": args.model, "overall": payload["overall"], "evidence_label": payload["evidence_label"]}, ensure_ascii=False, indent=2), flush=True)
    return payload


def summarize_round(round_name: str) -> dict[str, Any]:
    out_dir = RESULT_ROOT / round_name
    summaries = []
    scalar = Counter()
    evidence = Counter()
    controls = []
    groups = []
    for model_name in MODELS:
        path = out_dir / f"phase918_{model_name}_summary.json"
        if not path.exists():
            continue
        summary = read_json(path)
        summaries.append(summary)
        evidence[str(summary.get("evidence_label"))] += 1
        overall = summary.get("overall") or {}
        for key in [
            "rows",
            "boundary_rows",
            "channel_rows",
            "boundary_top1",
            "boundary_top5",
            "boundary_margin_nonnegative",
            "channel_top1",
            "channel_top5",
            "channel_top10",
            "channel_margin_nonnegative",
            "channel_promoted_margin",
            "channel_promoted_top1",
            "channel_promoted_top5",
            "channel_rank_improved",
            "weak_channel_candidate",
            "channel_blocker_suppressed",
            "channel_strict_clean_candidate",
        ]:
            scalar[key] += int(overall.get(key) or 0)
        scalar["selected_phase915_l39_candidates"] += int(summary.get("selected_phase915_l39_candidates") or 0)
        for row in summary.get("control_summaries") or []:
            item = dict(row)
            item["model"] = summary.get("model")
            controls.append(item)
        for row in summary.get("group_summaries") or []:
            item = dict(row)
            item["model"] = summary.get("model")
            groups.append(item)
    controls.sort(
        key=lambda row: (
            row.get("channel_strict_clean_candidate") or 0,
            row.get("channel_promoted_top1") or 0,
            row.get("channel_top1") or 0,
            row.get("channel_promoted_margin") or 0,
            row.get("channel_margin_nonnegative") or 0,
            row.get("channel_promoted_top5") or 0,
            row.get("weak_channel_candidate") or 0,
            row.get("channel_rank_improved") or 0,
            row.get("channel_blocker_suppressed") or 0,
            row.get("median_channel_margin_delta") or -9999,
        ),
        reverse=True,
    )
    groups.sort(
        key=lambda row: (
            row.get("channel_strict_clean_candidate") or 0,
            row.get("channel_top1") or 0,
            row.get("channel_margin_nonnegative") or 0,
            row.get("channel_promoted_top5") or 0,
            row.get("weak_channel_candidate") or 0,
            row.get("channel_rank_improved") or 0,
            row.get("channel_blocker_suppressed") or 0,
            row.get("median_channel_margin_delta") or -9999,
        ),
        reverse=True,
    )
    payload = {
        "phase": PHASE,
        "round": round_name,
        "status": "complete" if len(summaries) == len(MODELS) else "partial",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "models": [summary.get("model") for summary in summaries],
        "overall_scalar": {key: int(value) for key, value in sorted(scalar.items())},
        "evidence_label_counts": dict(sorted(evidence.items())),
        "model_summaries": summaries,
        "top_controls": controls[:160],
        "top_groups": groups[:80],
    }
    p846.write_json(out_dir / "phase918_cross_model_summary.json", payload)
    write_markdown(out_dir / "phase918_cross_model_summary.md", payload)
    return payload


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Phase 918 L39 MLP channel a-blocker suppressor localization",
        "",
        "## Overall",
        "",
        f"- models: {', '.join(payload.get('models') or [])}",
    ]
    for key, value in (payload.get("overall_scalar") or {}).items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Model Summaries", ""])
    lines.append(
        "| model | selected | rows | channel rows | channel top1 | channel margin>=0 | promoted margin | promoted top5 | weak channel | blocker suppressed | strict | evidence |"
    )
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
    for summary in payload.get("model_summaries") or []:
        overall = summary.get("overall") or {}
        lines.append(
            "| {model} | {selected} | {rows} | {channel_rows} | {top1} | {margin} | {prom_margin} | {prom_top5} | {weak} | {suppressed} | {strict} | {evidence} |".format(
                model=summary.get("model"),
                selected=summary.get("selected_phase915_l39_candidates"),
                rows=overall.get("rows"),
                channel_rows=overall.get("channel_rows"),
                top1=overall.get("channel_top1"),
                margin=overall.get("channel_margin_nonnegative"),
                prom_margin=overall.get("channel_promoted_margin"),
                prom_top5=overall.get("channel_promoted_top5"),
                weak=overall.get("weak_channel_candidate"),
                suppressed=overall.get("channel_blocker_suppressed"),
                strict=overall.get("channel_strict_clean_candidate"),
                evidence=summary.get("evidence_label"),
            )
        )
    lines.extend(["", "## Top Controls", ""])
    lines.append(
        "| model | control | family | group | factor | rows | top1 | margin>=0 | promoted margin | promoted top5 | weak | rank improved | blocker suppressed | median margin delta | mean eos delta | median blocker delta |"
    )
    lines.append("| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in payload.get("top_controls") or []:
        lines.append(
            "| {model} | {control_label} | {control_family} | {group_kind} | {factor} | {rows} | {channel_top1} | {channel_margin_nonnegative} | {channel_promoted_margin} | {channel_promoted_top5} | {weak_channel_candidate} | {channel_rank_improved} | {channel_blocker_suppressed} | {median_channel_margin_delta} | {mean_channel_eos_delta} | {median_channel_blocker_delta} |".format(
                **row
            )
        )
    lines.extend(["", "## Top Groups", ""])
    lines.append(
        "| model | group | family | rows | top1 | margin>=0 | promoted top5 | weak | rank improved | blocker suppressed | median margin delta |"
    )
    lines.append("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in payload.get("top_groups") or []:
        lines.append(
            "| {model} | {group_kind} | {control_family} | {rows} | {channel_top1} | {channel_margin_nonnegative} | {channel_promoted_top5} | {weak_channel_candidate} | {channel_rank_improved} | {channel_blocker_suppressed} | {median_channel_margin_delta} |".format(
                **row
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--round-name", default="l39_mlp_channel_a_blocker_suppressor_localization")
    parser.add_argument("--phase915-round", default="near_boundary_action_gate_search")
    parser.add_argument("--source-control-label", default="L39_mlp_output_scale_1.5")
    parser.add_argument("--boundary-blocker-token", default="a")
    parser.add_argument("--max-candidates-per-model", type=int, default=12)
    parser.add_argument("--target-layer", type=int, default=39)
    parser.add_argument("--max-prefix-tokens", type=int, default=5)
    parser.add_argument("--scale-up-factor", type=float, default=2.0)
    parser.add_argument("--l4-candidate-pool", type=int, default=512)
    parser.add_argument("--channel-candidate-pool", type=int, default=768)
    parser.add_argument("--band-size", type=int, default=32)
    parser.add_argument("--up-groups", default="eos_support_32,eos_support_64,margin_support_pos_32,margin_support_pos_64")
    parser.add_argument("--down-groups", default="a_blocker_support_32,a_blocker_support_64,a_logit_support_64,margin_support_neg_32,margin_support_neg_64,band_blocker_support_64")
    parser.add_argument("--general-groups", default="top_abs_64,low_abs_64")
    parser.add_argument("--up-factors", default="1.25,1.5,2.0")
    parser.add_argument("--down-factors", default="0.0,0.25,0.5,0.75")
    parser.add_argument("--general-factors", default="0.0,0.5,1.5,2.0")
    parser.add_argument("--log-every", type=int, default=2)
    parser.add_argument("--attn-implementations", default="flash_attention_2,sdpa")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summarize-round", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.summarize_round:
        payload = summarize_round(args.round_name)
        print(json.dumps({"phase": PHASE, "status": payload["status"], "overall": payload["overall_scalar"]}, ensure_ascii=False, indent=2))
        return
    if not args.model:
        raise SystemExit("--model is required unless --summarize-round is set")
    eval_model(args)


if __name__ == "__main__":
    main()
