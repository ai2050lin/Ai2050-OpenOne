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
from model_utils import get_layers  # noqa: E402


PHASE = 915
MODELS = ["qwen3", "glm4", "deepseek7b"]
PHASE914_ROOT = Path("tests/result/phase914_l4_mlp_route_near_holdout_validation")
RESULT_ROOT = Path("tests/result/phase915_near_boundary_action_gate_search")


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
        row.get("group_kind"),
        row.get("factor"),
    )


def score_phase914_row(row: dict[str, Any]) -> tuple[int, int, int, float, float]:
    promoted_top5 = int(bool(row.get("patched_eos_top5") and row.get("route_eos_rank") and int(row["route_eos_rank"]) > 5))
    weak = int(bool(row.get("weak_holdout_candidate")))
    rank_gain = 0
    if row.get("eos_rank_delta_vs_route") is not None:
        rank_gain = max(0, -int(row["eos_rank_delta_vs_route"]))
    margin = float(row.get("patched_eos_margin_vs_blocker") or -9999.0)
    band_drop = -float(row.get("band16_mean_logit_delta") or 9999.0)
    return promoted_top5, weak, rank_gain, margin, band_drop


def select_phase914_candidates(model_name: str, args: argparse.Namespace) -> list[dict[str, Any]]:
    path = PHASE914_ROOT / args.phase914_round / f"phase914_{model_name}_rows.jsonl"
    rows = read_jsonl(path)
    boundary_factors = {float(x) for x in parse_floats(args.boundary_factors)}
    selected = []
    for row in rows:
        if row.get("control_kind") != "mlp_channel_group_scale":
            continue
        if not row.get("route_near_top50"):
            continue
        if str(row.get("group_kind")) != "top_abs_64":
            continue
        if row.get("factor") is None or float(row.get("factor")) not in boundary_factors:
            continue
        promoted = bool(row.get("patched_eos_top5") and row.get("route_eos_rank") and int(row["route_eos_rank"]) > 5)
        if not (promoted or row.get("weak_holdout_candidate")):
            continue
        item = dict(row)
        item["phase914_promoted_top5_from_non_top5"] = promoted
        selected.append(item)
    buckets: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in selected:
        key = candidate_key(row)
        if key not in buckets or score_phase914_row(row) > score_phase914_row(buckets[key]):
            buckets[key] = row
    out = list(buckets.values())
    out.sort(
        key=lambda row: (
            score_phase914_row(row),
            str(row.get("eval_domain")),
            str(row.get("case_id")),
            str(row.get("prompt_variant")),
        ),
        reverse=True,
    )
    return out[: max(0, int(args.max_candidates_per_model))]


def resolve_layer_token(token: str, n_layers: int) -> int | None:
    token = str(token).strip()
    if token.startswith("L-"):
        return max(0, int(n_layers) - int(token[2:]))
    if token.startswith("L"):
        token = token[1:]
    try:
        layer = int(token)
    except ValueError:
        return None
    if layer < 0:
        layer = int(n_layers) + layer
    if 0 <= layer < int(n_layers):
        return layer
    return None


def parse_action_site(raw: str, n_layers: int) -> dict[str, Any] | None:
    raw = str(raw).strip()
    if raw == "l0_output":
        return {"site_label": "l0_output", "site_kind": "l0_output", "layer_idx": 0, "component": "attention"}
    if ":" not in raw:
        return None
    layer_token, component = raw.split(":", 1)
    component = component.strip()
    if component not in {"mlp", "attn"}:
        return None
    layer = resolve_layer_token(layer_token.strip(), n_layers)
    if layer is None:
        return None
    return {
        "site_label": f"L{layer}_{component}",
        "site_kind": "component_output",
        "layer_idx": int(layer),
        "component": component,
    }


def action_specs(n_layers: int, args: argparse.Namespace) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = [
        {
            "control_label": "boundary_precondition_only",
            "control_family": "boundary_precondition",
            "control_kind": "boundary_only",
            "neural_intervention": True,
            "diagnostic_only": False,
        }
    ]
    for k in [1, 3, 8]:
        specs.append(
            {
                "control_label": f"diagnostic_mask_boundary_blocker_top{k}",
                "control_family": "logit_mask_diagnostic",
                "control_kind": "logit_mask",
                "mask_topk_blockers": int(k),
                "neural_intervention": False,
                "diagnostic_only": True,
            }
        )
    sites = [site for site in (parse_action_site(raw, n_layers) for raw in parse_csv(args.action_sites)) if site]
    directions = parse_csv(args.direction_kinds)
    betas = parse_floats(args.betas)
    for site in sites:
        for direction in directions:
            for beta in betas:
                specs.append(
                    {
                        "control_label": f"{site['site_label']}_{direction}_beta_{beta:g}",
                        "control_family": "readout_action_vector",
                        "control_kind": "readout_action_vector",
                        "direction_kind": direction,
                        "beta": float(beta),
                        "neural_intervention": True,
                        "diagnostic_only": False,
                        **site,
                    }
                )
    for site in sites:
        if site["site_kind"] == "l0_output":
            continue
        for scale in parse_floats(args.component_scales):
            specs.append(
                {
                    "control_label": f"{site['site_label']}_output_scale_{scale:g}",
                    "control_family": "component_output_scale",
                    "control_kind": "component_output_scale",
                    "component_scale": float(scale),
                    "neural_intervention": True,
                    "diagnostic_only": False,
                    **site,
                }
            )
    return specs


def patch_tensor_last_token(tensor: torch.Tensor, vector: torch.Tensor | None = None, scale: float | None = None) -> torch.Tensor:
    patched = tensor.clone()
    if scale is not None:
        if patched.ndim >= 3:
            patched[:, -1, :] *= float(scale)
        elif patched.ndim >= 2:
            patched[-1, :] *= float(scale)
        else:
            patched *= float(scale)
        return patched
    if vector is None:
        return patched
    local = vector.to(device=patched.device, dtype=patched.dtype)
    if patched.ndim >= 3:
        patched[:, -1, :] += local
    elif patched.ndim >= 2:
        patched[-1, :] += local
    else:
        patched += local
    return patched


def patch_module_output(output, vector: torch.Tensor | None = None, scale: float | None = None):
    if torch.is_tensor(output):
        return patch_tensor_last_token(output, vector=vector, scale=scale)
    if isinstance(output, tuple) and output and torch.is_tensor(output[0]):
        return (patch_tensor_last_token(output[0], vector=vector, scale=scale), *output[1:])
    return output


def component_module(model, layer_idx: int, component: str):
    layers = get_layers(model)
    if not (0 <= int(layer_idx) < len(layers)):
        return None
    layer = layers[int(layer_idx)]
    if component == "mlp":
        return getattr(layer, "mlp", None)
    if component == "attn":
        return getattr(layer, "self_attn", None)
    return None


def install_action_hook(model, spec: dict[str, Any], vector: torch.Tensor | None) -> list[Any]:
    kind = spec.get("control_kind")
    if kind == "readout_action_vector":
        beta = float(spec.get("beta") or 0.0)
        if vector is None:
            return []
        action_vec = vector.float() * beta
        if spec.get("site_kind") == "l0_output":
            return p911.install_l0_output_vector(model, action_vec)
        module = component_module(model, int(spec.get("layer_idx") or 0), str(spec.get("component")))
        if module is None:
            return []
        return [module.register_forward_hook(lambda _module, _inputs, output: patch_module_output(output, vector=action_vec))]
    if kind == "component_output_scale":
        module = component_module(model, int(spec.get("layer_idx") or 0), str(spec.get("component")))
        if module is None:
            return []
        scale = float(spec.get("component_scale") or 1.0)
        return [module.register_forward_hook(lambda _module, _inputs, output: patch_module_output(output, scale=scale))]
    return []


def logits_with_boundary_and_action(
    model,
    device: torch.device,
    current_ids: list[int],
    route_delta: torch.Tensor,
    boundary_spec: dict[str, Any],
    action_spec: dict[str, Any],
    prompt_len: int,
    prefix_len: int,
    mlp_groups: dict[str, list[int]],
    action_vector: torch.Tensor | None,
) -> torch.Tensor | None:
    handles = p913.install_route_and_disentangle_hooks(
        model,
        route_delta,
        boundary_spec,
        prompt_len,
        prefix_len,
        len(current_ids),
        mlp_groups,
    )
    if action_spec.get("control_kind") not in {"boundary_only", "logit_mask"}:
        handles.extend(install_action_hook(model, action_spec, action_vector))
    if not handles:
        return None
    try:
        return p903.logits_plain(model, device, current_ids)
    finally:
        for handle in handles:
            handle.remove()


def finite_margin(metrics: dict[str, Any], blocker: dict[str, Any] | None) -> float | None:
    return p911.eos_margin_vs_blocker(metrics, blocker)


def token_logit(logits: torch.Tensor, token_id: int | None) -> float | None:
    return p911.token_logit(logits, token_id)


def row_from_logits(
    tokenizer,
    case: dict[str, Any],
    source_row: dict[str, Any],
    action_spec: dict[str, Any],
    prefix_ids: list[int],
    prefix_text: str,
    route_metrics: dict[str, Any],
    boundary_metrics: dict[str, Any],
    patched_metrics: dict[str, Any],
    route_logits: torch.Tensor,
    boundary_logits: torch.Tensor,
    patched_logits: torch.Tensor,
    route_top_rows: list[dict[str, Any]],
    boundary_top_rows: list[dict[str, Any]],
    patched_top_rows: list[dict[str, Any]],
    boundary_blocker_ids: list[int],
    route_delta_norm: float,
    action_vector_norm: float | None,
) -> dict[str, Any]:
    boundary_blocker = p910.first_non_eos_top(boundary_top_rows)
    patched_blocker = p910.first_non_eos_top(patched_top_rows)
    route_blocker = p910.first_non_eos_top(route_top_rows)
    boundary_rank = boundary_metrics.get("eos_rank")
    patched_rank = patched_metrics.get("eos_rank")
    boundary_eos_logit = boundary_metrics.get("eos_best_logit")
    patched_eos_logit = patched_metrics.get("eos_best_logit")
    boundary_margin = finite_margin(boundary_metrics, boundary_blocker)
    patched_margin = finite_margin(patched_metrics, patched_blocker)
    boundary_route_blocker_logit = token_logit(boundary_logits, boundary_blocker.get("token_id") if boundary_blocker else None)
    patched_route_blocker_logit = token_logit(patched_logits, boundary_blocker.get("token_id") if boundary_blocker else None)
    eos_top1 = bool(patched_rank == 1)
    rank_delta = None if boundary_rank is None or patched_rank is None else int(patched_rank) - int(boundary_rank)
    margin_delta = None if boundary_margin is None or patched_margin is None else float(patched_margin - boundary_margin)
    eos_delta = None if boundary_eos_logit is None or patched_eos_logit is None else float(patched_eos_logit - boundary_eos_logit)
    route_band16 = p912.stats_for_ids(boundary_logits, boundary_blocker_ids[:16])
    patched_band16 = p912.stats_for_ids(patched_logits, boundary_blocker_ids[:16])
    band16_delta = None if route_band16["mean"] is None or patched_band16["mean"] is None else patched_band16["mean"] - route_band16["mean"]
    return {
        "phase": PHASE,
        "row_kind": "phase915_near_boundary_action_gate_row",
        "model": source_row.get("model"),
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
        "boundary_group_kind": source_row.get("group_kind"),
        "boundary_factor": source_row.get("factor"),
        "phase914_weak_holdout_candidate": bool(source_row.get("weak_holdout_candidate")),
        "phase914_promoted_top5_from_non_top5": bool(source_row.get("phase914_promoted_top5_from_non_top5")),
        "control_label": action_spec.get("control_label"),
        "control_family": action_spec.get("control_family"),
        "control_kind": action_spec.get("control_kind"),
        "direction_kind": action_spec.get("direction_kind"),
        "beta": action_spec.get("beta"),
        "site_kind": action_spec.get("site_kind"),
        "site_label": action_spec.get("site_label"),
        "layer_idx": action_spec.get("layer_idx"),
        "component": action_spec.get("component"),
        "component_scale": action_spec.get("component_scale"),
        "mask_topk_blockers": action_spec.get("mask_topk_blockers"),
        "neural_intervention": bool(action_spec.get("neural_intervention")),
        "diagnostic_only": bool(action_spec.get("diagnostic_only")),
        "prompt_input_intact": True,
        "prompt_all_zero_used_as_test_control": False,
        "route_delta_norm": route_delta_norm,
        "action_vector_norm": action_vector_norm,
        "route_eos_rank": route_metrics.get("eos_rank"),
        "route_blocker_token": route_blocker.get("token") if route_blocker else None,
        "boundary_eos_rank": boundary_rank,
        "boundary_eos_logit": boundary_eos_logit,
        "boundary_eos_top5": bool(boundary_rank is not None and int(boundary_rank) <= 5),
        "boundary_eos_margin_vs_blocker": boundary_margin,
        "boundary_blocker_token": boundary_blocker.get("token") if boundary_blocker else None,
        "boundary_blocker_logit": boundary_blocker.get("logit") if boundary_blocker else None,
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
        "boundary_blocker_logit_after_patch": patched_route_blocker_logit,
        "boundary_blocker_logit_delta": None
        if boundary_route_blocker_logit is None or patched_route_blocker_logit is None
        else float(patched_route_blocker_logit - boundary_route_blocker_logit),
        "boundary_band16_mean_delta": band16_delta,
        "promoted_margin_from_negative": bool(
            boundary_margin is not None and boundary_margin < 0 and patched_margin is not None and patched_margin >= 0
        ),
        "promoted_top1_from_non_top1": bool(boundary_rank is not None and int(boundary_rank) > 1 and patched_rank == 1),
        "promoted_top5_from_non_top5": bool(
            boundary_rank is not None and int(boundary_rank) > 5 and patched_rank is not None and int(patched_rank) <= 5
        ),
        "rank_improved": bool(rank_delta is not None and rank_delta < 0),
        "weak_action_candidate": bool(
            rank_delta is not None
            and rank_delta < 0
            and eos_delta is not None
            and eos_delta >= 0
            and margin_delta is not None
            and margin_delta > 0
        ),
        "strict_clean_candidate": p911.strict_clean_candidate(tokenizer, case, prefix_ids, eos_top1),
        "boundary_top8": boundary_top_rows[:8],
        "patched_top8": patched_top_rows[:8],
    }


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    action_rows = [row for row in rows if row.get("neural_intervention") and row.get("control_kind") != "boundary_only"]
    diagnostic_rows = [row for row in rows if row.get("diagnostic_only")]
    boundary_rows = [row for row in rows if row.get("control_kind") == "boundary_only"]
    return {
        "rows": len(rows),
        "boundary_rows": len(boundary_rows),
        "action_rows": len(action_rows),
        "diagnostic_rows": len(diagnostic_rows),
        "boundary_top1": sum(1 for row in boundary_rows if row.get("patched_eos_top1")),
        "boundary_top5": sum(1 for row in boundary_rows if row.get("patched_eos_top5")),
        "boundary_margin_nonnegative": sum(1 for row in boundary_rows if row.get("patched_eos_margin_nonnegative")),
        "action_top1": sum(1 for row in action_rows if row.get("patched_eos_top1")),
        "action_top5": sum(1 for row in action_rows if row.get("patched_eos_top5")),
        "action_top10": sum(1 for row in action_rows if row.get("patched_eos_top10")),
        "action_margin_nonnegative": sum(1 for row in action_rows if row.get("patched_eos_margin_nonnegative")),
        "action_promoted_margin": sum(1 for row in action_rows if row.get("promoted_margin_from_negative")),
        "action_promoted_top1": sum(1 for row in action_rows if row.get("promoted_top1_from_non_top1")),
        "action_promoted_top5": sum(1 for row in action_rows if row.get("promoted_top5_from_non_top5")),
        "action_rank_improved": sum(1 for row in action_rows if row.get("rank_improved")),
        "weak_action_candidate": sum(1 for row in action_rows if row.get("weak_action_candidate")),
        "action_strict_clean_candidate": sum(1 for row in action_rows if row.get("strict_clean_candidate")),
        "diagnostic_top1": sum(1 for row in diagnostic_rows if row.get("patched_eos_top1")),
        "diagnostic_margin_nonnegative": sum(1 for row in diagnostic_rows if row.get("patched_eos_margin_nonnegative")),
        "diagnostic_promoted_margin": sum(1 for row in diagnostic_rows if row.get("promoted_margin_from_negative")),
        "median_action_margin_delta": median([row.get("eos_margin_delta_vs_boundary") for row in action_rows]),
        "median_action_rank_delta": median([row.get("eos_rank_delta_vs_boundary") for row in action_rows]),
        "mean_action_margin_delta": mean([row.get("eos_margin_delta_vs_boundary") for row in action_rows]),
        "mean_action_eos_delta": mean([row.get("eos_logit_delta_vs_boundary") for row in action_rows]),
        "boundary_blocker_tokens_top12": dict(Counter(str(row.get("boundary_blocker_token")) for row in rows).most_common(12)),
        "patched_blocker_tokens_top12": dict(Counter(str(row.get("patched_blocker_token")) for row in action_rows).most_common(12)),
    }


def summarize_by_control(rows: list[dict[str, Any]], limit: int = 80) -> list[dict[str, Any]]:
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
                "direction_kind": first.get("direction_kind"),
                "site_label": first.get("site_label"),
                "beta": first.get("beta"),
                "component_scale": first.get("component_scale"),
                "diagnostic_only": first.get("diagnostic_only"),
            }
        )
        out.append(summary)
    out.sort(
        key=lambda row: (
            row.get("action_strict_clean_candidate") or 0,
            row.get("action_promoted_top1") or 0,
            row.get("action_top1") or 0,
            row.get("action_promoted_margin") or 0,
            row.get("action_margin_nonnegative") or 0,
            row.get("action_promoted_top5") or 0,
            row.get("weak_action_candidate") or 0,
            row.get("action_rank_improved") or 0,
            row.get("diagnostic_margin_nonnegative") or 0,
            row.get("median_action_margin_delta") or -9999,
        ),
        reverse=True,
    )
    return out[:limit]


def summarize_model(model_name: str, rows: list[dict[str, Any]], selected_count: int, specs_count: int, attn_impl: str | None) -> dict[str, Any]:
    overall = summarize_rows(rows)
    if selected_count == 0:
        evidence = "no_phase914_near_boundary_candidates"
    elif overall["action_strict_clean_candidate"] > 0:
        evidence = "action_gate_strict_clean_candidate_found"
    elif overall["action_promoted_top1"] > 0 or overall["action_top1"] > 0:
        evidence = "action_gate_top1_candidate_found"
    elif overall["action_promoted_margin"] > 0 or overall["action_margin_nonnegative"] > 0:
        evidence = "action_gate_margin_candidate_found"
    elif overall["diagnostic_promoted_margin"] > 0:
        evidence = "diagnostic_blocker_mask_can_close_margin"
    elif overall["action_promoted_top5"] > 0 or overall["weak_action_candidate"] > 0:
        evidence = "partial_action_rank_candidate_only"
    else:
        evidence = "no_action_gate_candidate_found"
    return {
        "phase": PHASE,
        "title": "Near-boundary Action Gate Search After L4 MLP Boundary Adjustment",
        "model": model_name,
        "status": "complete",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "attn_implementation": attn_impl,
        "selected_phase914_candidates": selected_count,
        "action_spec_count": specs_count,
        "overall": overall,
        "control_summaries": summarize_by_control(rows),
        "evidence_label": evidence,
        "boundary": (
            "Phase915 fixes the Phase914 GLM4 L4 top_abs_64 boundary precondition and searches "
            "for a downstream action gate. Logit masks are diagnostic upper bounds only."
        ),
    }


def eval_model(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = RESULT_ROOT / args.round_name
    out_dir.mkdir(parents=True, exist_ok=True)
    selected = select_phase914_candidates(args.model, args)
    if args.dry_run or not selected:
        empty_summary = summarize_model(args.model, [], len(selected), 0, None)
        payload = {
            "phase": PHASE,
            "model": args.model,
            "status": "dry_run" if selected else "no_phase914_candidates",
            "selected_phase914_candidates": len(selected),
            "preview": selected[:20],
            "overall": empty_summary["overall"],
            "control_summaries": [],
            "evidence_label": empty_summary["evidence_label"],
        }
        p846.write_json(out_dir / f"phase915_{args.model}_summary.json", payload)
        p846.write_jsonl(out_dir / f"phase915_{args.model}_rows.jsonl", [])
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
        lm_weight = p911.output_embedding_weight(model)
        all_specs = action_specs(len(get_layers(model)), args)
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
            route_logits, mlp_activation = p913.capture_route_logits_and_mlp_activation(model, device, period_ids, route_delta, 4)
            if route_logits is None:
                continue
            route_metrics = p903.state_metrics(tokenizer, route_logits, groups)
            route_top_rows = p910.topk_tokens(tokenizer, route_logits, groups, max(64, int(args.band_size)))
            route_band32_ids = p911.top_non_eos_ids(route_top_rows, int(args.band_size))
            route_band16_ids = route_band32_ids[: min(16, len(route_band32_ids))]
            mlp_groups, _mlp_diag = p913.mlp_channel_groups_for_case(
                model,
                device,
                mlp_activation,
                route_metrics.get("eos_best_id"),
                route_band16_ids,
                route_band32_ids,
                int(args.mlp_candidate_pool),
            )
            boundary_spec = {
                "control_label": f"L4_mlp_channels_top_abs_64_scale_{float(source_row.get('factor')):g}",
                "control_kind": "mlp_channel_group_scale",
                "layer_idx": 4,
                "group_kind": "top_abs_64",
                "factor": float(source_row.get("factor")),
            }
            boundary_logits = p913.logits_with_spec(
                model,
                device,
                period_ids,
                route_delta,
                boundary_spec,
                len(prompt_ids),
                len(prefix_ids),
                mlp_groups,
            )
            if boundary_logits is None:
                continue
            boundary_metrics = p903.state_metrics(tokenizer, boundary_logits, groups)
            boundary_top_rows = p910.topk_tokens(tokenizer, boundary_logits, groups, max(64, int(args.band_size)))
            boundary_blocker_ids = p911.top_non_eos_ids(boundary_top_rows, int(args.band_size))
            for spec in all_specs:
                action_vector = None
                vector_norm = None
                if spec.get("control_kind") == "readout_action_vector":
                    action_vector = p911.readout_direction(
                        lm_weight,
                        int(route_delta.numel()),
                        boundary_metrics.get("eos_best_id"),
                        boundary_blocker_ids,
                        str(spec.get("direction_kind")),
                    )
                    if action_vector is None:
                        continue
                    vector_norm = float(torch.linalg.vector_norm(action_vector.float()).item())
                if spec.get("control_kind") == "boundary_only":
                    patched_logits = boundary_logits
                elif spec.get("control_kind") == "logit_mask":
                    patched_logits = p911.masked_logits(boundary_logits, boundary_blocker_ids[: int(spec.get("mask_topk_blockers") or 0)])
                else:
                    patched_logits = logits_with_boundary_and_action(
                        model,
                        device,
                        period_ids,
                        route_delta,
                        boundary_spec,
                        spec,
                        len(prompt_ids),
                        len(prefix_ids),
                        mlp_groups,
                        action_vector,
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
                        route_logits,
                        boundary_logits,
                        patched_logits,
                        route_top_rows,
                        boundary_top_rows,
                        patched_top_rows,
                        boundary_blocker_ids,
                        route_delta_norm,
                        vector_norm,
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
    p846.write_json(out_dir / f"phase915_{args.model}_summary.json", payload)
    p846.write_jsonl(out_dir / f"phase915_{args.model}_rows.jsonl", rows)
    print(json.dumps({"phase": PHASE, "model": args.model, "overall": payload["overall"], "evidence_label": payload["evidence_label"]}, ensure_ascii=False, indent=2), flush=True)
    return payload


def summarize_round(round_name: str) -> dict[str, Any]:
    out_dir = RESULT_ROOT / round_name
    summaries = []
    scalar = Counter()
    evidence = Counter()
    controls = []
    for model_name in MODELS:
        path = out_dir / f"phase915_{model_name}_summary.json"
        if not path.exists():
            continue
        summary = read_json(path)
        summaries.append(summary)
        evidence[str(summary.get("evidence_label"))] += 1
        overall = summary.get("overall") or {}
        for key in [
            "rows",
            "boundary_rows",
            "action_rows",
            "diagnostic_rows",
            "boundary_top1",
            "boundary_top5",
            "boundary_margin_nonnegative",
            "action_top1",
            "action_top5",
            "action_top10",
            "action_margin_nonnegative",
            "action_promoted_margin",
            "action_promoted_top1",
            "action_promoted_top5",
            "action_rank_improved",
            "weak_action_candidate",
            "action_strict_clean_candidate",
            "diagnostic_top1",
            "diagnostic_margin_nonnegative",
            "diagnostic_promoted_margin",
        ]:
            scalar[key] += int(overall.get(key) or 0)
        scalar["selected_phase914_candidates"] += int(summary.get("selected_phase914_candidates") or 0)
        for row in summary.get("control_summaries") or []:
            item = dict(row)
            item["model"] = summary.get("model")
            controls.append(item)
    controls.sort(
        key=lambda row: (
            row.get("action_strict_clean_candidate") or 0,
            row.get("action_promoted_top1") or 0,
            row.get("action_top1") or 0,
            row.get("action_promoted_margin") or 0,
            row.get("action_margin_nonnegative") or 0,
            row.get("action_promoted_top5") or 0,
            row.get("weak_action_candidate") or 0,
            row.get("action_rank_improved") or 0,
            row.get("diagnostic_margin_nonnegative") or 0,
            row.get("median_action_margin_delta") or -9999,
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
        "top_controls": controls[:120],
    }
    p846.write_json(out_dir / "phase915_cross_model_summary.json", payload)
    write_markdown(out_dir / "phase915_cross_model_summary.md", payload)
    return payload


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Phase 915 near-boundary action gate search",
        "",
        "## Overall",
        "",
        f"- models: {', '.join(payload.get('models') or [])}",
    ]
    for key, value in (payload.get("overall_scalar") or {}).items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Model Summaries", ""])
    lines.append(
        "| model | selected | rows | action rows | action top1 | action margin>=0 | promoted margin | promoted top5 | weak action | strict | diagnostic margin | evidence |"
    )
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
    for summary in payload.get("model_summaries") or []:
        overall = summary.get("overall") or {}
        lines.append(
            "| {model} | {selected} | {rows} | {action_rows} | {top1} | {margin} | {prom_margin} | {prom_top5} | {weak} | {strict} | {diag_margin} | {evidence} |".format(
                model=summary.get("model"),
                selected=summary.get("selected_phase914_candidates"),
                rows=overall.get("rows"),
                action_rows=overall.get("action_rows"),
                top1=overall.get("action_top1"),
                margin=overall.get("action_margin_nonnegative"),
                prom_margin=overall.get("action_promoted_margin"),
                prom_top5=overall.get("action_promoted_top5"),
                weak=overall.get("weak_action_candidate"),
                strict=overall.get("action_strict_clean_candidate"),
                diag_margin=overall.get("diagnostic_margin_nonnegative"),
                evidence=summary.get("evidence_label"),
            )
        )
    lines.extend(["", "## Top Controls", ""])
    lines.append(
        "| model | control | family | site | direction | beta | scale | rows | top1 | margin>=0 | promoted margin | promoted top5 | weak | rank improved | median margin delta | mean eos delta |"
    )
    lines.append("| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in payload.get("top_controls") or []:
        lines.append(
            "| {model} | {control_label} | {control_family} | {site_label} | {direction_kind} | {beta} | {component_scale} | {rows} | {action_top1} | {action_margin_nonnegative} | {action_promoted_margin} | {action_promoted_top5} | {weak_action_candidate} | {action_rank_improved} | {median_action_margin_delta} | {mean_action_eos_delta} |".format(
                **row
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--round-name", default="near_boundary_action_gate_search")
    parser.add_argument("--phase914-round", default="l4_mlp_route_near_holdout_validation")
    parser.add_argument("--max-candidates-per-model", type=int, default=12)
    parser.add_argument("--boundary-factors", default="0.3,0.4")
    parser.add_argument("--max-prefix-tokens", type=int, default=5)
    parser.add_argument("--scale-up-factor", type=float, default=2.0)
    parser.add_argument("--mlp-candidate-pool", type=int, default=512)
    parser.add_argument("--band-size", type=int, default=32)
    parser.add_argument("--action-sites", default="l0_output,L-1:mlp,L-1:attn,L-4:mlp,L-4:attn")
    parser.add_argument("--direction-kinds", default="eos_minus_blocker_top1,minus_blocker_top1,minus_blocker_top3_mean,eos_boost")
    parser.add_argument("--betas", default="0.05,0.1,0.25,0.5")
    parser.add_argument("--component-scales", default="0.0,0.5,1.5")
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
