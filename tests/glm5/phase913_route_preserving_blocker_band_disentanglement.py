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
import phase906_eos_action_boundary_test as p906  # noqa: E402
import phase909_l0_attention_source_span_eos_boundary_audit as p909  # noqa: E402
import phase910_prompt_preserving_termination_route_reconstruction as p910  # noqa: E402
import phase911_full_vocab_blocker_displacement_audit as p911  # noqa: E402
import phase912_finite_blocker_band_source_localization as p912  # noqa: E402
from model_utils import get_layers  # noqa: E402


PHASE = 913
MODELS = ["qwen3", "glm4", "deepseek7b"]
RESULT_ROOT = Path("tests/result/phase913_route_preserving_blocker_band_disentanglement")
PHASE899_ROUND = "domain_axis_rollout_protocol_audit"


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def parse_factors(raw: str) -> list[float]:
    out = []
    for part in str(raw).split(","):
        part = part.strip()
        if part:
            out.append(float(part))
    return out or [0.75, 0.5, 0.25]


def median(values: list[float | int | None]) -> float | None:
    cleaned = [float(value) for value in values if value is not None]
    return None if not cleaned else float(statistics.median(cleaned))


def mean(values: list[float | int | None]) -> float | None:
    cleaned = [float(value) for value in values if value is not None]
    return None if not cleaned else float(sum(cleaned) / len(cleaned))


def attention_head_count(model, layer_idx: int) -> int:
    layer = get_layers(model)[int(layer_idx)]
    attn = getattr(layer, "self_attn", None)
    for name in ["num_heads", "num_attention_heads", "n_heads"]:
        value = getattr(attn, name, None)
        if value:
            return int(value)
    cfg = getattr(model, "config", None)
    for name in ["num_attention_heads", "n_head", "num_heads"]:
        value = getattr(cfg, name, None)
        if value:
            return int(value)
    o_proj = getattr(attn, "o_proj", None)
    q_proj = getattr(attn, "q_proj", None)
    if o_proj is not None and q_proj is not None:
        return int(getattr(q_proj, "out_features", o_proj.in_features) // max(1, getattr(attn, "head_dim", 128)))
    return 0


def head_dim_for(model, layer_idx: int, n_heads: int) -> int:
    layer = get_layers(model)[int(layer_idx)]
    attn = getattr(layer, "self_attn", None)
    value = getattr(attn, "head_dim", None)
    if value:
        return int(value)
    o_proj = getattr(attn, "o_proj", None)
    if o_proj is not None and n_heads > 0:
        return int(o_proj.in_features // n_heads)
    cfg = getattr(model, "config", None)
    hidden = getattr(cfg, "hidden_size", None)
    if hidden and n_heads > 0:
        return int(hidden // n_heads)
    return 0


def base_specs(model, factors: list[float], span_kinds: list[str], mlp_group_kinds: list[str]) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = [
        {
            "control_label": "route_only_alpha_1",
            "control_family": "route_control",
            "control_kind": "route_only",
            "subunit_family": "route_only",
            "factor": None,
        }
    ]
    n_heads = attention_head_count(model, 0)
    for factor in factors:
        suffix = f"scale_{factor:g}"
        for head_idx in range(n_heads):
            specs.append(
                {
                    "control_label": f"L0_attention_head_{head_idx}_{suffix}",
                    "control_family": "subunit_disentanglement",
                    "control_kind": "attention_head_scale",
                    "subunit_family": "l0_attention_head",
                    "layer_idx": 0,
                    "head_idx": int(head_idx),
                    "factor": float(factor),
                }
            )
        for span_kind in span_kinds:
            specs.append(
                {
                    "control_label": f"L0_attention_span_{span_kind}_{suffix}",
                    "control_family": "subunit_disentanglement",
                    "control_kind": "attention_span_scale",
                    "subunit_family": "l0_attention_span",
                    "layer_idx": 0,
                    "span_kind": span_kind,
                    "factor": float(factor),
                }
            )
        for group_kind in mlp_group_kinds:
            specs.append(
                {
                    "control_label": f"L4_mlp_channels_{group_kind}_{suffix}",
                    "control_family": "subunit_disentanglement",
                    "control_kind": "mlp_channel_group_scale",
                    "subunit_family": "l4_mlp_channel_group",
                    "layer_idx": 4,
                    "group_kind": group_kind,
                    "factor": float(factor),
                }
            )
    return specs


def install_attention_head_scale(model, layer_idx: int, head_idx: int, factor: float) -> list[Any]:
    layer = get_layers(model)[int(layer_idx)]
    attn = getattr(layer, "self_attn", None)
    o_proj = getattr(attn, "o_proj", None)
    if o_proj is None:
        return []
    n_heads = attention_head_count(model, int(layer_idx))
    head_dim = head_dim_for(model, int(layer_idx), n_heads)
    if n_heads <= 0 or head_dim <= 0:
        return []
    start = int(head_idx) * int(head_dim)
    end = min(start + int(head_dim), int(o_proj.in_features))
    if start >= end:
        return []

    def scale_hidden(hidden_states: torch.Tensor) -> torch.Tensor | None:
        if not torch.is_tensor(hidden_states):
            return None
        patched = hidden_states.clone()
        if patched.ndim >= 3:
            patched[:, -1, start:end] *= float(factor)
        elif patched.ndim >= 2:
            patched[-1, start:end] *= float(factor)
        return patched

    def hook_with_kwargs(_module, inputs, kwargs):
        if kwargs and torch.is_tensor(kwargs.get("input")):
            patched = scale_hidden(kwargs["input"])
            if patched is None:
                return None
            new_kwargs = dict(kwargs)
            new_kwargs["input"] = patched
            return inputs, new_kwargs
        if inputs and torch.is_tensor(inputs[0]):
            patched = scale_hidden(inputs[0])
            if patched is None:
                return None
            return (patched, *inputs[1:]), kwargs
        return None

    def hook_positional(_module, inputs):
        if inputs and torch.is_tensor(inputs[0]):
            patched = scale_hidden(inputs[0])
            if patched is None:
                return None
            return (patched, *inputs[1:])
        return None

    try:
        return [o_proj.register_forward_pre_hook(hook_with_kwargs, with_kwargs=True)]
    except TypeError:
        return [o_proj.register_forward_pre_hook(hook_positional)]


def mlp_down_proj(model, layer_idx: int):
    layer = get_layers(model)[int(layer_idx)]
    mlp = getattr(layer, "mlp", None)
    return getattr(mlp, "down_proj", None)


def install_mlp_channel_group_scale(model, layer_idx: int, channel_ids: list[int], factor: float) -> list[Any]:
    down_proj = mlp_down_proj(model, int(layer_idx))
    if down_proj is None or not channel_ids:
        return []
    idx = torch.tensor(sorted(set(int(x) for x in channel_ids)), dtype=torch.long)

    def scale_hidden(hidden_states: torch.Tensor) -> torch.Tensor | None:
        if not torch.is_tensor(hidden_states):
            return None
        if hidden_states.shape[-1] <= int(idx.max().item()):
            return None
        local_idx = idx.to(hidden_states.device)
        patched = hidden_states.clone()
        if patched.ndim >= 3:
            patched[:, -1, local_idx] *= float(factor)
        elif patched.ndim >= 2:
            patched[-1, local_idx] *= float(factor)
        return patched

    def hook_with_kwargs(_module, inputs, kwargs):
        if kwargs and torch.is_tensor(kwargs.get("input")):
            patched = scale_hidden(kwargs["input"])
            if patched is None:
                return None
            new_kwargs = dict(kwargs)
            new_kwargs["input"] = patched
            return inputs, new_kwargs
        if inputs and torch.is_tensor(inputs[0]):
            patched = scale_hidden(inputs[0])
            if patched is None:
                return None
            return (patched, *inputs[1:]), kwargs
        return None

    def hook_positional(_module, inputs):
        if inputs and torch.is_tensor(inputs[0]):
            patched = scale_hidden(inputs[0])
            if patched is None:
                return None
            return (patched, *inputs[1:])
        return None

    try:
        return [down_proj.register_forward_pre_hook(hook_with_kwargs, with_kwargs=True)]
    except TypeError:
        return [down_proj.register_forward_pre_hook(hook_positional)]


def capture_route_logits_and_mlp_activation(
    model,
    device: torch.device,
    current_ids: list[int],
    route_delta: torch.Tensor,
    mlp_layer_idx: int,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    handles = p911.install_l0_output_vector(model, route_delta)
    down_proj = mlp_down_proj(model, int(mlp_layer_idx))
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
    try:
        logits = p903.logits_plain(model, device, current_ids)
    finally:
        for handle in handles:
            handle.remove()
    return logits, captured.get("activation")


def lm_head_rows(model, token_ids: list[int], device: torch.device) -> torch.Tensor | None:
    weight = None
    if hasattr(model, "lm_head"):
        weight = model.lm_head.weight
    elif hasattr(model, "embed_out"):
        weight = model.embed_out.weight
    if weight is None or getattr(weight, "is_meta", False):
        return None
    valid = [int(x) for x in token_ids if 0 <= int(x) < int(weight.shape[0])]
    if not valid:
        return None
    idx = torch.tensor(valid, dtype=torch.long, device=weight.device)
    return weight.index_select(0, idx).detach().to(device=device, dtype=torch.float32)


def mlp_channel_groups_for_case(
    model,
    device: torch.device,
    activation: torch.Tensor | None,
    eos_id: int | None,
    band16_ids: list[int],
    band32_ids: list[int],
    candidate_pool: int,
) -> tuple[dict[str, list[int]], dict[str, Any]]:
    if activation is None or eos_id is None:
        return {}, {}
    down_proj = mlp_down_proj(model, 4)
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
        "top_abs_32": [int(x) for x in top_abs[: min(32, len(top_abs))].tolist()],
        "top_abs_64": [int(x) for x in top_abs[: min(64, len(top_abs))].tolist()],
        "low_abs_64": [int(x) for x in low_abs[: min(64, len(low_abs))].tolist()],
    }
    token_ids = [int(eos_id)] + [int(x) for x in band16_ids] + [int(x) for x in band32_ids]
    token_rows = lm_head_rows(model, token_ids, device)
    diagnostics: dict[str, Any] = {
        "activation_abs_top": float(abs_vals[top_abs[0]].item()) if len(top_abs) else None,
        "activation_abs_median": float(torch.median(abs_vals).item()),
    }
    if token_rows is None:
        return groups, diagnostics
    eos_row = token_rows[0:1]
    band16_count = len(band16_ids)
    band32_count = len(band32_ids)
    band16_rows = token_rows[1 : 1 + band16_count]
    band32_rows = token_rows[1 + band16_count : 1 + band16_count + band32_count]
    candidate_idx = top_abs.to(device=down_proj.weight.device)
    down_cols = down_proj.weight.index_select(1, candidate_idx).detach().to(device=device, dtype=torch.float32)
    act_sub = act.index_select(0, top_abs).to(device=device, dtype=torch.float32)
    eos_proj = torch.matmul(eos_row, down_cols).squeeze(0)
    band16_proj = torch.matmul(band16_rows, down_cols).mean(dim=0) if band16_rows.numel() else eos_proj * 0
    band32_proj = torch.matmul(band32_rows, down_cols).mean(dim=0) if band32_rows.numel() else eos_proj * 0
    support16 = act_sub * (band16_proj - eos_proj)
    support32 = act_sub * (band32_proj - eos_proj)
    for name, support, budget in [
        ("band16_support_32", support16, 32),
        ("band16_support_64", support16, 64),
        ("band32_support_64", support32, 64),
    ]:
        k = min(int(budget), int(support.numel()))
        if k <= 0:
            continue
        top_support = torch.topk(support, k=k, largest=True).indices
        chosen = candidate_idx.index_select(0, top_support).detach().cpu().tolist()
        groups[name] = [int(x) for x in chosen]
        diagnostics[f"{name}_mean_support"] = float(support.index_select(0, top_support).mean().item())
        diagnostics[f"{name}_max_support"] = float(support.index_select(0, top_support[:1]).max().item())
    return groups, diagnostics


def install_route_and_disentangle_hooks(
    model,
    route_delta: torch.Tensor,
    spec: dict[str, Any],
    prompt_len: int,
    prefix_len: int,
    seq_len: int,
    mlp_groups: dict[str, list[int]],
) -> list[Any]:
    handles = p911.install_l0_output_vector(model, route_delta)
    kind = spec.get("control_kind")
    factor = float(spec.get("factor") or 1.0)
    if kind == "attention_head_scale":
        handles.extend(install_attention_head_scale(model, int(spec.get("layer_idx") or 0), int(spec.get("head_idx") or 0), factor))
    elif kind == "attention_span_scale":
        span_start, span_end = p909.span_bounds(str(spec.get("span_kind")), int(prompt_len), int(prefix_len), int(seq_len))
        handles.extend(p909.install_attention_input_span_scale(model, int(spec.get("layer_idx") or 0), span_start, span_end, factor))
    elif kind == "mlp_channel_group_scale":
        group = mlp_groups.get(str(spec.get("group_kind"))) or []
        handles.extend(install_mlp_channel_group_scale(model, int(spec.get("layer_idx") or 4), group, factor))
    return handles


def logits_with_spec(
    model,
    device: torch.device,
    current_ids: list[int],
    route_delta: torch.Tensor,
    spec: dict[str, Any],
    prompt_len: int,
    prefix_len: int,
    mlp_groups: dict[str, list[int]],
) -> torch.Tensor | None:
    handles = install_route_and_disentangle_hooks(
        model,
        route_delta,
        spec,
        prompt_len,
        prefix_len,
        len(current_ids),
        mlp_groups,
    )
    if not handles:
        return None
    try:
        return p903.logits_plain(model, device, current_ids)
    finally:
        for handle in handles:
            handle.remove()


def stats_for_ids(logits: torch.Tensor, token_ids: list[int]) -> dict[str, Any]:
    return p912.stats_for_ids(logits, token_ids)


def make_row(
    tokenizer,
    source_row: dict[str, Any],
    case: dict[str, Any],
    spec: dict[str, Any],
    prefix_ids: list[int],
    prefix_text: str,
    route_metrics: dict[str, Any],
    patched_metrics: dict[str, Any],
    route_logits: torch.Tensor,
    patched_logits: torch.Tensor,
    route_top_rows: list[dict[str, Any]],
    patched_top_rows: list[dict[str, Any]],
    band16_ids: list[int],
    band32_ids: list[int],
    route_delta_norm: float,
    mlp_groups: dict[str, list[int]],
    mlp_diag: dict[str, Any],
) -> dict[str, Any]:
    route_band16 = stats_for_ids(route_logits, band16_ids)
    route_band32 = stats_for_ids(route_logits, band32_ids)
    patched_band16 = stats_for_ids(patched_logits, band16_ids)
    patched_band32 = stats_for_ids(patched_logits, band32_ids)
    eos_id = route_metrics.get("eos_best_id")
    route_eos_logit = route_metrics.get("eos_best_logit")
    patched_eos_logit = patched_metrics.get("eos_best_logit")
    patched_eos_rank = patched_metrics.get("eos_rank")
    route_eos_rank = route_metrics.get("eos_rank")
    patched_blocker = p910.first_non_eos_top(patched_top_rows)
    route_blocker = p910.first_non_eos_top(route_top_rows)
    patched_margin = p911.eos_margin_vs_blocker(patched_metrics, patched_blocker)
    route_margin = p911.eos_margin_vs_blocker(route_metrics, route_blocker)
    band16_mean_delta = None if patched_band16["mean"] is None or route_band16["mean"] is None else patched_band16["mean"] - route_band16["mean"]
    band32_mean_delta = None if patched_band32["mean"] is None or route_band32["mean"] is None else patched_band32["mean"] - route_band32["mean"]
    band16_max_delta = None if patched_band16["max"] is None or route_band16["max"] is None else patched_band16["max"] - route_band16["max"]
    band32_max_delta = None if patched_band32["max"] is None or route_band32["max"] is None else patched_band32["max"] - route_band32["max"]
    eos_delta = None if route_eos_logit is None or patched_eos_logit is None else float(patched_eos_logit - route_eos_logit)
    rank_delta = None if patched_eos_rank is None or route_eos_rank is None else int(patched_eos_rank) - int(route_eos_rank)
    group_kind = spec.get("group_kind")
    group_ids = mlp_groups.get(str(group_kind)) if group_kind else None
    eos_top1 = bool(patched_eos_rank == 1)
    route_preserving_disentangle = bool(
        band16_mean_delta is not None
        and band16_mean_delta <= -0.25
        and eos_delta is not None
        and eos_delta >= 0.0
    )
    strong_route_preserving_disentangle = bool(
        band16_mean_delta is not None
        and band16_mean_delta <= -0.5
        and eos_delta is not None
        and eos_delta >= 0.0
        and rank_delta is not None
        and rank_delta <= 0
    )
    return {
        "phase": PHASE,
        "row_kind": "phase913_route_preserving_blocker_band_disentanglement_row",
        "model": source_row.get("model"),
        "source_key": source_row.get("source_key"),
        "source_subset_key": source_row.get("source_subset_key"),
        "eval_domain": source_row.get("eval_domain"),
        "case_id": source_row.get("case_id"),
        "case_split": source_row.get("case_split"),
        "object": source_row.get("object"),
        "canonical_answer": source_row.get("canonical_answer"),
        "prompt_variant": source_row.get("prompt_variant"),
        "edit_mode": source_row.get("edit_mode"),
        "prefix_text": prefix_text,
        "control_label": spec.get("control_label"),
        "control_family": spec.get("control_family"),
        "control_kind": spec.get("control_kind"),
        "subunit_family": spec.get("subunit_family"),
        "layer_idx": spec.get("layer_idx"),
        "head_idx": spec.get("head_idx"),
        "span_kind": spec.get("span_kind"),
        "group_kind": group_kind,
        "factor": spec.get("factor"),
        "prompt_input_intact": True,
        "prompt_all_zero_used_as_test_control": False,
        "route_delta_norm": route_delta_norm,
        "route_eos_rank": route_eos_rank,
        "route_eos_logit": route_eos_logit,
        "route_blocker_token": route_blocker.get("token") if route_blocker else None,
        "route_eos_margin_vs_blocker": route_margin,
        "patched_eos_rank": patched_eos_rank,
        "patched_eos_logit": patched_eos_logit,
        "patched_eos_top1": eos_top1,
        "patched_eos_top5": bool(patched_eos_rank is not None and int(patched_eos_rank) <= 5),
        "patched_eos_top10": bool(patched_eos_rank is not None and int(patched_eos_rank) <= 10),
        "patched_eos_top50": bool(patched_eos_rank is not None and int(patched_eos_rank) <= 50),
        "patched_blocker_token": patched_blocker.get("token") if patched_blocker else None,
        "patched_eos_margin_vs_blocker": patched_margin,
        "patched_eos_margin_nonnegative": bool(patched_margin is not None and patched_margin >= 0),
        "eos_logit_delta_vs_route": eos_delta,
        "eos_rank_delta_vs_route": rank_delta,
        "band16_ids": band16_ids,
        "band16_tokens": [p903.decode_token(tokenizer, token_id) for token_id in band16_ids],
        "band32_ids": band32_ids,
        "band32_tokens": [p903.decode_token(tokenizer, token_id) for token_id in band32_ids],
        "route_band16_mean_logit": route_band16["mean"],
        "route_band16_max_logit": route_band16["max"],
        "patched_band16_mean_logit": patched_band16["mean"],
        "patched_band16_max_logit": patched_band16["max"],
        "route_band32_mean_logit": route_band32["mean"],
        "route_band32_max_logit": route_band32["max"],
        "patched_band32_mean_logit": patched_band32["mean"],
        "patched_band32_max_logit": patched_band32["max"],
        "band16_mean_logit_delta": band16_mean_delta,
        "band32_mean_logit_delta": band32_mean_delta,
        "band16_max_logit_delta": band16_max_delta,
        "band32_max_logit_delta": band32_max_delta,
        "route_preserving_disentangle_candidate": route_preserving_disentangle,
        "strong_route_preserving_disentangle_candidate": strong_route_preserving_disentangle,
        "strict_clean_candidate": p911.strict_clean_candidate(tokenizer, case, prefix_ids, eos_top1),
        "mlp_group_size": len(group_ids or []),
        "mlp_group_preview": [int(x) for x in (group_ids or [])[:16]],
        "mlp_diag": mlp_diag if spec.get("subunit_family") == "l4_mlp_channel_group" else {},
        "route_top8": route_top_rows[:8],
        "patched_top8": patched_top_rows[:8],
    }


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    source_rows = [row for row in rows if row.get("control_kind") != "route_only"]
    route_rows = [row for row in rows if row.get("control_kind") == "route_only"]
    return {
        "rows": len(rows),
        "source_rows": len(source_rows),
        "route_rows": len(route_rows),
        "route_eos_top10": sum(1 for row in route_rows if row.get("patched_eos_top10")),
        "route_eos_top50": sum(1 for row in route_rows if row.get("patched_eos_top50")),
        "source_eos_top1": sum(1 for row in source_rows if row.get("patched_eos_top1")),
        "source_eos_top5": sum(1 for row in source_rows if row.get("patched_eos_top5")),
        "source_eos_top10": sum(1 for row in source_rows if row.get("patched_eos_top10")),
        "source_eos_top50": sum(1 for row in source_rows if row.get("patched_eos_top50")),
        "source_margin_nonnegative": sum(1 for row in source_rows if row.get("patched_eos_margin_nonnegative")),
        "strict_clean_candidate": sum(1 for row in rows if row.get("strict_clean_candidate")),
        "source_strict_clean_candidate": sum(1 for row in source_rows if row.get("strict_clean_candidate")),
        "route_preserving_disentangle_candidate": sum(1 for row in source_rows if row.get("route_preserving_disentangle_candidate")),
        "strong_route_preserving_disentangle_candidate": sum(1 for row in source_rows if row.get("strong_route_preserving_disentangle_candidate")),
        "median_band16_mean_delta": median([row.get("band16_mean_logit_delta") for row in source_rows]),
        "median_band32_mean_delta": median([row.get("band32_mean_logit_delta") for row in source_rows]),
        "median_eos_logit_delta": median([row.get("eos_logit_delta_vs_route") for row in source_rows]),
        "mean_eos_logit_delta": mean([row.get("eos_logit_delta_vs_route") for row in source_rows]),
        "route_blocker_tokens_top12": dict(Counter(str(row.get("route_blocker_token")) for row in rows).most_common(12)),
        "patched_blocker_tokens_top12": dict(Counter(str(row.get("patched_blocker_token")) for row in rows).most_common(12)),
    }


def summarize_by_spec(rows: list[dict[str, Any]], top_n: int = 120) -> list[dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("control_kind") == "route_only":
            continue
        buckets[str(row.get("control_label"))].append(row)
    out = []
    for _label, vals in buckets.items():
        summary = summarize_rows(vals)
        first = vals[0]
        summary.update(
            {
                "control_label": first.get("control_label"),
                "control_kind": first.get("control_kind"),
                "subunit_family": first.get("subunit_family"),
                "layer_idx": first.get("layer_idx"),
                "head_idx": first.get("head_idx"),
                "span_kind": first.get("span_kind"),
                "group_kind": first.get("group_kind"),
                "factor": first.get("factor"),
            }
        )
        out.append(summary)
    out.sort(
        key=lambda row: (
            row.get("source_strict_clean_candidate") or 0,
            row.get("source_eos_top1") or 0,
            row.get("source_margin_nonnegative") or 0,
            row.get("source_eos_top5") or 0,
            row.get("strong_route_preserving_disentangle_candidate") or 0,
            row.get("route_preserving_disentangle_candidate") or 0,
            -(row.get("median_band16_mean_delta") or 9999),
            row.get("median_eos_logit_delta") or -9999,
        ),
        reverse=True,
    )
    return out[:top_n]


def summarize_by_family(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, float], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("control_kind") == "route_only":
            continue
        buckets[(str(row.get("subunit_family")), float(row.get("factor") or 0.0))].append(row)
    out = []
    for (family, factor), vals in buckets.items():
        summary = summarize_rows(vals)
        summary.update({"subunit_family": family, "factor": factor})
        out.append(summary)
    out.sort(
        key=lambda row: (
            row.get("source_eos_top1") or 0,
            row.get("source_margin_nonnegative") or 0,
            row.get("source_eos_top5") or 0,
            row.get("strong_route_preserving_disentangle_candidate") or 0,
            row.get("route_preserving_disentangle_candidate") or 0,
        ),
        reverse=True,
    )
    return out


def summarize_model(model_name: str, rows: list[dict[str, Any]], selected_count: int, attn_impl: str | None) -> dict[str, Any]:
    overall = summarize_rows(rows)
    if overall["source_eos_top1"] > 0:
        evidence = "subunit_disentanglement_reaches_eos_top1"
    elif overall["source_margin_nonnegative"] > 0:
        evidence = "subunit_disentanglement_crosses_margin"
    elif overall["source_eos_top5"] > 0 and overall["strong_route_preserving_disentangle_candidate"] > 0:
        evidence = "route_preserving_disentangle_reaches_eos_top5"
    elif overall["strong_route_preserving_disentangle_candidate"] > 0:
        evidence = "strong_route_preserving_disentangle_candidates_found"
    elif overall["route_preserving_disentangle_candidate"] > 0:
        evidence = "weak_route_preserving_disentangle_candidates_found"
    elif overall["route_eos_top50"] > 0:
        evidence = "route_near_but_no_disentangled_subunit_found"
    else:
        evidence = "no_route_near_for_disentanglement"
    return {
        "phase": PHASE,
        "title": "Route-preserving Blocker Band Disentanglement",
        "model": model_name,
        "status": "complete",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "attn_implementation": attn_impl,
        "selected_answer_drift_rows": selected_count,
        "overall": overall,
        "spec_summaries": summarize_by_spec(rows),
        "family_summaries": summarize_by_family(rows),
        "evidence_label": evidence,
        "boundary": (
            "Phase913 uses mild subunit interventions on L0 attention heads/spans and L4 MLP channel groups. "
            "A positive source candidate must lower the blocker band while preserving or increasing EOS logit; "
            "full component zeroing is not used as closure evidence."
        ),
    }


def eval_model(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = RESULT_ROOT / args.round_name
    out_dir.mkdir(parents=True, exist_ok=True)
    selected_rows = p906.selected_phase899_rows(args.model, args)
    if args.dry_run or not selected_rows:
        payload = {
            "phase": PHASE,
            "model": args.model,
            "status": "dry_run" if selected_rows else "no_rows",
            "selected_rows": len(selected_rows),
        }
        p846.write_json(out_dir / f"phase913_{args.model}_summary.json", payload)
        p846.write_jsonl(out_dir / f"phase913_{args.model}_rows.jsonl", [])
        print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
        return payload
    factors = parse_factors(args.factors)
    span_kinds = [x.strip() for x in str(args.span_kinds).split(",") if x.strip()]
    mlp_group_kinds = [x.strip() for x in str(args.mlp_group_kinds).split(",") if x.strip()]
    case_map = {str(case.get("case_id")): case for case in p885.extended_cases()}
    rows: list[dict[str, Any]] = []
    model = None
    tokenizer = None
    attn_impl = None
    try:
        model, tokenizer, device, attn_impl = p862.p844.p828.p796.load_model_bf16_prefer_flash(
            args.model, args.attn_implementations
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        specs = base_specs(model, factors, span_kinds, mlp_group_kinds)
        groups = p903.protocol_category_groups(tokenizer)
        prompt_cache: dict[tuple[str, str], list[int]] = {}
        for idx, source_row in enumerate(selected_rows, 1):
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
            baseline_logits, base_vec = p910.logits_and_l0_vector(model, device, period_ids)
            prompt_zero_handles = p909.install_attention_input_span_scale(model, 0, 0, len(prompt_ids), 0.0)
            _prompt_zero_logits, prompt_zero_vec = p910.logits_and_l0_vector(model, device, period_ids, prompt_zero_handles)
            if base_vec is None or prompt_zero_vec is None:
                continue
            route_delta = prompt_zero_vec - base_vec
            route_delta_norm = float(torch.linalg.vector_norm(route_delta).item())
            if route_delta_norm <= 0:
                continue
            route_logits, mlp_activation = capture_route_logits_and_mlp_activation(model, device, period_ids, route_delta, 4)
            if route_logits is None:
                continue
            route_metrics = p903.state_metrics(tokenizer, route_logits, groups)
            route_top_rows = p910.topk_tokens(tokenizer, route_logits, groups, max(64, int(args.band_size)))
            band32_ids = p911.top_non_eos_ids(route_top_rows, int(args.band_size))
            band16_ids = band32_ids[: min(16, len(band32_ids))]
            mlp_groups, mlp_diag = mlp_channel_groups_for_case(
                model,
                device,
                mlp_activation,
                route_metrics.get("eos_best_id"),
                band16_ids,
                band32_ids,
                int(args.mlp_candidate_pool),
            )
            for spec in specs:
                if spec.get("control_kind") == "route_only":
                    patched_logits = route_logits
                else:
                    if spec.get("control_kind") == "mlp_channel_group_scale" and not mlp_groups.get(str(spec.get("group_kind"))):
                        continue
                    patched_logits = logits_with_spec(
                        model,
                        device,
                        period_ids,
                        route_delta,
                        spec,
                        len(prompt_ids),
                        len(prefix_ids),
                        mlp_groups,
                    )
                    if patched_logits is None:
                        continue
                patched_metrics = p903.state_metrics(tokenizer, patched_logits, groups)
                patched_top_rows = p910.topk_tokens(tokenizer, patched_logits, groups, 16)
                rows.append(
                    make_row(
                        tokenizer,
                        source_row,
                        case,
                        spec,
                        prefix_ids,
                        prefix_text,
                        route_metrics,
                        patched_metrics,
                        route_logits,
                        patched_logits,
                        route_top_rows,
                        patched_top_rows,
                        band16_ids,
                        band32_ids,
                        route_delta_norm,
                        mlp_groups,
                        mlp_diag,
                    )
                )
            if idx % max(1, int(args.log_every)) == 0 or idx == len(selected_rows):
                log(f"{args.model}/{args.round_name}: row={idx}/{len(selected_rows)} rows={len(rows)} specs={len(specs)}")
    finally:
        if model is not None:
            p862.p844.p828.release_model(model)
        if tokenizer is not None:
            del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    payload = summarize_model(args.model, rows, len(selected_rows), attn_impl)
    p846.write_json(out_dir / f"phase913_{args.model}_summary.json", payload)
    p846.write_jsonl(out_dir / f"phase913_{args.model}_rows.jsonl", rows)
    print(json.dumps({"phase": PHASE, "model": args.model, "overall": payload["overall"], "evidence_label": payload["evidence_label"]}, ensure_ascii=False, indent=2), flush=True)
    return payload


def summarize_round(round_name: str) -> dict[str, Any]:
    out_dir = RESULT_ROOT / round_name
    summaries = []
    scalar = Counter()
    evidence = Counter()
    top_specs = []
    top_families = []
    for model_name in MODELS:
        path = out_dir / f"phase913_{model_name}_summary.json"
        if not path.exists():
            continue
        summary = read_json(path)
        summaries.append(summary)
        evidence[str(summary.get("evidence_label"))] += 1
        overall = summary.get("overall") or {}
        for key in [
            "rows",
            "source_rows",
            "route_rows",
            "route_eos_top10",
            "route_eos_top50",
            "source_eos_top1",
            "source_eos_top5",
            "source_eos_top10",
            "source_eos_top50",
            "source_margin_nonnegative",
            "strict_clean_candidate",
            "source_strict_clean_candidate",
            "route_preserving_disentangle_candidate",
            "strong_route_preserving_disentangle_candidate",
        ]:
            scalar[key] += int(overall.get(key) or 0)
        for row in summary.get("spec_summaries") or []:
            item = dict(row)
            item["model"] = summary.get("model")
            top_specs.append(item)
        for row in summary.get("family_summaries") or []:
            item = dict(row)
            item["model"] = summary.get("model")
            top_families.append(item)
    top_specs.sort(
        key=lambda row: (
            row.get("source_strict_clean_candidate") or 0,
            row.get("source_eos_top1") or 0,
            row.get("source_margin_nonnegative") or 0,
            row.get("source_eos_top5") or 0,
            row.get("strong_route_preserving_disentangle_candidate") or 0,
            row.get("route_preserving_disentangle_candidate") or 0,
            -(row.get("median_band16_mean_delta") or 9999),
            row.get("median_eos_logit_delta") or -9999,
        ),
        reverse=True,
    )
    top_families.sort(
        key=lambda row: (
            row.get("source_eos_top1") or 0,
            row.get("source_margin_nonnegative") or 0,
            row.get("source_eos_top5") or 0,
            row.get("strong_route_preserving_disentangle_candidate") or 0,
            row.get("route_preserving_disentangle_candidate") or 0,
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
        "top_specs": top_specs[:120],
        "top_families": top_families[:60],
    }
    p846.write_json(out_dir / "phase913_cross_model_summary.json", payload)
    write_markdown(out_dir / "phase913_cross_model_summary.md", payload)
    return payload


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Phase 913 route-preserving blocker band disentanglement",
        "",
        "## Overall",
        "",
        f"- models: {', '.join(payload.get('models') or [])}",
    ]
    for key, value in (payload.get("overall_scalar") or {}).items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Model Summaries", ""])
    lines.append(
        "| model | rows | source rows | route top10 | route top50 | source top1 | source top5 | margin>=0 | weak disentangle | strong disentangle | evidence |"
    )
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
    for summary in payload.get("model_summaries") or []:
        overall = summary.get("overall") or {}
        lines.append(
            "| {model} | {rows} | {source_rows} | {route10} | {route50} | {top1} | {top5} | {margin} | {weak} | {strong} | {evidence} |".format(
                model=summary.get("model"),
                rows=overall.get("rows"),
                source_rows=overall.get("source_rows"),
                route10=overall.get("route_eos_top10"),
                route50=overall.get("route_eos_top50"),
                top1=overall.get("source_eos_top1"),
                top5=overall.get("source_eos_top5"),
                margin=overall.get("source_margin_nonnegative"),
                weak=overall.get("route_preserving_disentangle_candidate"),
                strong=overall.get("strong_route_preserving_disentangle_candidate"),
                evidence=summary.get("evidence_label"),
            )
        )
    lines.extend(["", "## Top Specs", ""])
    lines.append(
        "| model | family | label | factor | rows | top1 | top5 | margin>=0 | weak | strong | median band16 delta | median eos delta | blockers |"
    )
    lines.append("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
    for row in payload.get("top_specs") or []:
        lines.append(
            "| {model} | {subunit_family} | {control_label} | {factor} | {rows} | {source_eos_top1} | {source_eos_top5} | {source_margin_nonnegative} | {route_preserving_disentangle_candidate} | {strong_route_preserving_disentangle_candidate} | {median_band16_mean_delta} | {median_eos_logit_delta} | {route_blocker_tokens_top12} |".format(
                **row
            )
        )
    lines.extend(["", "## Top Families", ""])
    lines.append(
        "| model | family | factor | rows | top1 | top5 | margin>=0 | weak | strong | median band16 delta | median eos delta |"
    )
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in payload.get("top_families") or []:
        lines.append(
            "| {model} | {subunit_family} | {factor} | {rows} | {source_eos_top1} | {source_eos_top5} | {source_margin_nonnegative} | {route_preserving_disentangle_candidate} | {strong_route_preserving_disentangle_candidate} | {median_band16_mean_delta} | {median_eos_logit_delta} |".format(
                **row
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--round-name", default="route_preserving_blocker_band_disentanglement")
    parser.add_argument("--phase899-round", default=PHASE899_ROUND)
    parser.add_argument("--max-rows-per-model", type=int, default=0)
    parser.add_argument("--max-prefix-tokens", type=int, default=5)
    parser.add_argument("--scale-up-factor", type=float, default=2.0)
    parser.add_argument("--factors", default="0.75,0.5,0.25")
    parser.add_argument("--band-size", type=int, default=32)
    parser.add_argument("--span-kinds", default="prompt_all,prompt_first8,prompt_last8,answer_prefix_all,last8_before_period,period_token")
    parser.add_argument("--mlp-group-kinds", default="band16_support_32,band16_support_64,band32_support_64,top_abs_64,low_abs_64")
    parser.add_argument("--mlp-candidate-pool", type=int, default=512)
    parser.add_argument("--log-every", type=int, default=4)
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
