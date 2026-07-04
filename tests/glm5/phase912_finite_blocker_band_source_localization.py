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
from model_utils import get_layers  # noqa: E402


PHASE = 912
MODELS = ["qwen3", "glm4", "deepseek7b"]
RESULT_ROOT = Path("tests/result/phase912_finite_blocker_band_source_localization")
PHASE899_ROUND = "domain_axis_rollout_protocol_audit"


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def median(values: list[float | int | None]) -> float | None:
    cleaned = [float(value) for value in values if value is not None]
    return None if not cleaned else float(statistics.median(cleaned))


def mean(values: list[float | int | None]) -> float | None:
    cleaned = [float(value) for value in values if value is not None]
    return None if not cleaned else float(sum(cleaned) / len(cleaned))


def factors_from_arg(raw: str) -> list[float]:
    out = []
    for part in str(raw).split(","):
        part = part.strip()
        if part:
            out.append(float(part))
    return out or [0.5, 0.0]


def source_specs(model, layer_stride: int, factors: list[float]) -> list[dict[str, Any]]:
    layers = get_layers(model)
    stride = max(1, int(layer_stride))
    indices = list(range(0, len(layers), stride))
    if len(layers) - 1 not in indices:
        indices.append(len(layers) - 1)
    out = [
        {
            "control_label": "route_only_alpha_1",
            "control_family": "route_control",
            "control_kind": "route_only",
            "layer_idx": None,
            "component_kind": None,
            "factor": None,
        }
    ]
    for factor in factors:
        suffix = "zero" if float(factor) == 0.0 else f"scale_{factor:g}"
        for layer_idx in indices:
            for component_kind in ["attention", "mlp"]:
                out.append(
                    {
                        "control_label": f"L{layer_idx}_{component_kind}_{suffix}",
                        "control_family": "component_source_suppression",
                        "control_kind": "component_scale",
                        "layer_idx": int(layer_idx),
                        "component_kind": component_kind,
                        "factor": float(factor),
                    }
                )
    return out


def scale_component_output(output, factor: float):
    if isinstance(output, tuple):
        if not output or not torch.is_tensor(output[0]):
            return output
        patched = output[0].clone()
        if patched.ndim >= 3:
            patched[:, -1, :] *= float(factor)
        elif patched.ndim >= 2:
            patched[-1, :] *= float(factor)
        return (patched, *output[1:])
    if torch.is_tensor(output):
        patched = output.clone()
        if patched.ndim >= 3:
            patched[:, -1, :] *= float(factor)
        elif patched.ndim >= 2:
            patched[-1, :] *= float(factor)
        return patched
    return output


def install_route_and_source_hooks(
    model,
    route_delta: torch.Tensor,
    spec: dict[str, Any],
) -> list[Any]:
    handles: list[Any] = []
    route_module = p903.component_module(model, 0, "attention")
    target_layer = spec.get("layer_idx")
    target_component = spec.get("component_kind")
    factor = spec.get("factor")
    target_module = (
        p903.component_module(model, int(target_layer), str(target_component))
        if target_layer is not None and target_component is not None
        else None
    )

    if route_module is not None and target_module is route_module and factor is not None:
        def combined_hook(_module, _inputs, output):
            tensor = p910.attn_output_tensor(output)
            if tensor is None:
                return output
            patched = tensor.clone()
            local_delta = route_delta.to(device=patched.device, dtype=patched.dtype)
            if patched.ndim >= 3:
                patched[:, -1, :] += local_delta
                patched[:, -1, :] *= float(factor)
            elif patched.ndim >= 2:
                patched[-1, :] += local_delta
                patched[-1, :] *= float(factor)
            return p910.replace_attn_output(output, patched)

        return [route_module.register_forward_hook(combined_hook)]

    if route_module is not None:
        handles.extend(p911.install_l0_output_vector(model, route_delta))
    if target_module is not None and factor is not None:
        handles.append(target_module.register_forward_hook(lambda _m, _i, output: scale_component_output(output, float(factor))))
    return handles


def logits_with_route_and_source(
    model,
    device: torch.device,
    current_ids: list[int],
    route_delta: torch.Tensor,
    spec: dict[str, Any],
) -> torch.Tensor | None:
    handles = install_route_and_source_hooks(model, route_delta, spec)
    if not handles:
        return None
    try:
        return p903.logits_plain(model, device, current_ids)
    finally:
        for handle in handles:
            handle.remove()


def stats_for_ids(logits: torch.Tensor, token_ids: list[int]) -> dict[str, Any]:
    valid = [int(token_id) for token_id in token_ids if 0 <= int(token_id) < int(logits.numel())]
    if not valid:
        return {"max": None, "mean": None, "sum": None, "max_id": None}
    scores = [(token_id, float(logits[token_id].item())) for token_id in valid]
    max_id, max_score = max(scores, key=lambda item: item[1])
    values = [score for _token_id, score in scores]
    return {
        "max": float(max_score),
        "mean": float(sum(values) / len(values)),
        "sum": float(sum(values)),
        "max_id": int(max_id),
    }


def token_delta_map(logits: torch.Tensor, base_logits: torch.Tensor, token_ids: list[int]) -> dict[str, float]:
    out: dict[str, float] = {}
    for token_id in token_ids:
        token_id = int(token_id)
        if 0 <= token_id < int(logits.numel()):
            out[str(token_id)] = float(logits[token_id].item() - base_logits[token_id].item())
    return out


def layer_bucket(layer_idx: int | None, n_layers: int) -> str:
    if layer_idx is None:
        return "none"
    pos = (int(layer_idx) + 1) / max(1, int(n_layers))
    if pos <= 0.33:
        return "early"
    if pos <= 0.66:
        return "middle"
    return "late"


def make_row(
    tokenizer,
    groups: dict[str, list[int]],
    source_row: dict[str, Any],
    case: dict[str, Any],
    spec: dict[str, Any],
    n_layers: int,
    prefix_ids: list[int],
    prefix_text: str,
    baseline_metrics: dict[str, Any],
    route_metrics: dict[str, Any],
    patched_metrics: dict[str, Any],
    route_logits: torch.Tensor,
    patched_logits: torch.Tensor,
    route_top_rows: list[dict[str, Any]],
    patched_top_rows: list[dict[str, Any]],
    band16_ids: list[int],
    band32_ids: list[int],
    route_delta_norm: float,
) -> dict[str, Any]:
    route_band16 = stats_for_ids(route_logits, band16_ids)
    route_band32 = stats_for_ids(route_logits, band32_ids)
    patched_band16 = stats_for_ids(patched_logits, band16_ids)
    patched_band32 = stats_for_ids(patched_logits, band32_ids)
    eos_id = route_metrics.get("eos_best_id")
    route_eos_logit = route_metrics.get("eos_best_logit")
    patched_eos_logit = patched_metrics.get("eos_best_logit")
    patched_eos_rank = patched_metrics.get("eos_rank")
    patched_blocker = p910.first_non_eos_top(patched_top_rows)
    route_blocker = p910.first_non_eos_top(route_top_rows)
    patched_blocker_margin = p911.eos_margin_vs_blocker(patched_metrics, patched_blocker)
    route_blocker_margin = p911.eos_margin_vs_blocker(route_metrics, route_blocker)
    band16_max_delta = None if patched_band16["max"] is None or route_band16["max"] is None else patched_band16["max"] - route_band16["max"]
    band32_max_delta = None if patched_band32["max"] is None or route_band32["max"] is None else patched_band32["max"] - route_band32["max"]
    band16_mean_delta = None if patched_band16["mean"] is None or route_band16["mean"] is None else patched_band16["mean"] - route_band16["mean"]
    band32_mean_delta = None if patched_band32["mean"] is None or route_band32["mean"] is None else patched_band32["mean"] - route_band32["mean"]
    eos_logit_delta = None if route_eos_logit is None or patched_eos_logit is None else float(patched_eos_logit - route_eos_logit)
    eos_margin_vs_band16 = None if patched_eos_logit is None or patched_band16["max"] is None else float(patched_eos_logit - patched_band16["max"])
    eos_margin_vs_band32 = None if patched_eos_logit is None or patched_band32["max"] is None else float(patched_eos_logit - patched_band32["max"])
    layer_idx = spec.get("layer_idx")
    eos_top1 = bool(patched_eos_rank == 1)
    return {
        "phase": PHASE,
        "row_kind": "phase912_finite_blocker_band_source_localization_row",
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
        "layer_idx": layer_idx,
        "layer_bucket": layer_bucket(None if layer_idx is None else int(layer_idx), n_layers),
        "component_kind": spec.get("component_kind"),
        "factor": spec.get("factor"),
        "prompt_input_intact": True,
        "prompt_all_zero_used_as_test_control": False,
        "route_delta_norm": route_delta_norm,
        "baseline_eos_rank": baseline_metrics.get("eos_rank"),
        "route_eos_rank": route_metrics.get("eos_rank"),
        "route_eos_logit": route_eos_logit,
        "route_blocker_token": route_blocker.get("token") if route_blocker else None,
        "route_blocker_category": route_blocker.get("category") if route_blocker else None,
        "route_eos_margin_vs_blocker": route_blocker_margin,
        "patched_eos_rank": patched_eos_rank,
        "patched_eos_logit": patched_eos_logit,
        "patched_eos_top1": eos_top1,
        "patched_eos_top5": bool(patched_eos_rank is not None and int(patched_eos_rank) <= 5),
        "patched_eos_top10": bool(patched_eos_rank is not None and int(patched_eos_rank) <= 10),
        "patched_eos_top50": bool(patched_eos_rank is not None and int(patched_eos_rank) <= 50),
        "patched_blocker_token": patched_blocker.get("token") if patched_blocker else None,
        "patched_blocker_category": patched_blocker.get("category") if patched_blocker else None,
        "patched_eos_margin_vs_blocker": patched_blocker_margin,
        "patched_eos_margin_nonnegative": bool(patched_blocker_margin is not None and patched_blocker_margin >= 0),
        "eos_logit_delta_vs_route": eos_logit_delta,
        "eos_rank_delta_vs_route": None
        if patched_eos_rank is None or route_metrics.get("eos_rank") is None
        else int(patched_eos_rank) - int(route_metrics["eos_rank"]),
        "band16_ids": band16_ids,
        "band16_tokens": [p903.decode_token(tokenizer, token_id) for token_id in band16_ids],
        "band32_ids": band32_ids,
        "band32_tokens": [p903.decode_token(tokenizer, token_id) for token_id in band32_ids],
        "route_band16_max_logit": route_band16["max"],
        "route_band16_mean_logit": route_band16["mean"],
        "route_band16_max_token": p903.decode_token(tokenizer, route_band16["max_id"]),
        "route_band32_max_logit": route_band32["max"],
        "route_band32_mean_logit": route_band32["mean"],
        "route_band32_max_token": p903.decode_token(tokenizer, route_band32["max_id"]),
        "patched_band16_max_logit": patched_band16["max"],
        "patched_band16_mean_logit": patched_band16["mean"],
        "patched_band16_max_token": p903.decode_token(tokenizer, patched_band16["max_id"]),
        "patched_band32_max_logit": patched_band32["max"],
        "patched_band32_mean_logit": patched_band32["mean"],
        "patched_band32_max_token": p903.decode_token(tokenizer, patched_band32["max_id"]),
        "band16_max_logit_delta": band16_max_delta,
        "band16_mean_logit_delta": band16_mean_delta,
        "band32_max_logit_delta": band32_max_delta,
        "band32_mean_logit_delta": band32_mean_delta,
        "eos_margin_vs_band16": eos_margin_vs_band16,
        "eos_margin_vs_band32": eos_margin_vs_band32,
        "band16_source_candidate": bool(band16_mean_delta is not None and band16_mean_delta <= -0.5),
        "band32_source_candidate": bool(band32_mean_delta is not None and band32_mean_delta <= -0.5),
        "band16_strong_source_candidate": bool(band16_mean_delta is not None and band16_mean_delta <= -1.0),
        "band32_strong_source_candidate": bool(band32_mean_delta is not None and band32_mean_delta <= -1.0),
        "strict_clean_candidate": p911.strict_clean_candidate(tokenizer, case, prefix_ids, eos_top1),
        "band16_token_deltas": token_delta_map(patched_logits, route_logits, band16_ids),
        "route_top8": route_top_rows[:8],
        "patched_top8": patched_top_rows[:8],
    }


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    source_rows = [row for row in rows if row.get("control_kind") == "component_scale"]
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
        "band16_source_candidate": sum(1 for row in source_rows if row.get("band16_source_candidate")),
        "band32_source_candidate": sum(1 for row in source_rows if row.get("band32_source_candidate")),
        "band16_strong_source_candidate": sum(1 for row in source_rows if row.get("band16_strong_source_candidate")),
        "band32_strong_source_candidate": sum(1 for row in source_rows if row.get("band32_strong_source_candidate")),
        "median_band16_mean_delta": median([row.get("band16_mean_logit_delta") for row in source_rows]),
        "median_band32_mean_delta": median([row.get("band32_mean_logit_delta") for row in source_rows]),
        "median_band16_max_delta": median([row.get("band16_max_logit_delta") for row in source_rows]),
        "median_band32_max_delta": median([row.get("band32_max_logit_delta") for row in source_rows]),
        "median_eos_logit_delta": median([row.get("eos_logit_delta_vs_route") for row in source_rows]),
        "route_blocker_tokens_top12": dict(Counter(str(row.get("route_blocker_token")) for row in rows).most_common(12)),
        "patched_blocker_tokens_top12": dict(Counter(str(row.get("patched_blocker_token")) for row in rows).most_common(12)),
    }


def summarize_by_source(rows: list[dict[str, Any]], top_n: int = 120) -> list[dict[str, Any]]:
    buckets: dict[tuple[Any, Any, Any], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("control_kind") != "component_scale":
            continue
        buckets[(row.get("layer_idx"), row.get("component_kind"), row.get("factor"))].append(row)
    summaries = []
    for (layer_idx, component_kind, factor), vals in buckets.items():
        summary = summarize_rows(vals)
        first = vals[0]
        summary.update(
            {
                "layer_idx": layer_idx,
                "layer_bucket": first.get("layer_bucket"),
                "component_kind": component_kind,
                "factor": factor,
                "control_label": first.get("control_label"),
            }
        )
        summaries.append(summary)
    summaries.sort(
        key=lambda row: (
            row.get("source_strict_clean_candidate") or 0,
            row.get("source_eos_top1") or 0,
            row.get("source_eos_top5") or 0,
            row.get("source_margin_nonnegative") or 0,
            row.get("band16_strong_source_candidate") or 0,
            row.get("band16_source_candidate") or 0,
            -(row.get("median_band16_mean_delta") or 9999),
            -(row.get("median_band32_mean_delta") or 9999),
        ),
        reverse=True,
    )
    return summaries[:top_n]


def summarize_by_bucket(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, str, float], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("control_kind") != "component_scale":
            continue
        buckets[(str(row.get("layer_bucket")), str(row.get("component_kind")), float(row.get("factor")))].append(row)
    out = []
    for (bucket, component, factor), vals in buckets.items():
        summary = summarize_rows(vals)
        summary.update({"layer_bucket": bucket, "component_kind": component, "factor": factor})
        out.append(summary)
    out.sort(
        key=lambda row: (
            row.get("source_eos_top1") or 0,
            row.get("source_eos_top5") or 0,
            row.get("source_margin_nonnegative") or 0,
            -(row.get("median_band16_mean_delta") or 9999),
        ),
        reverse=True,
    )
    return out


def summarize_model(model_name: str, rows: list[dict[str, Any]], selected_count: int, attn_impl: str | None) -> dict[str, Any]:
    overall = summarize_rows(rows)
    source_summaries = summarize_by_source(rows)
    bucket_summaries = summarize_by_bucket(rows)
    if overall["source_eos_top1"] > 0:
        evidence = "component_source_scan_reaches_eos_top1"
    elif overall["source_eos_top5"] > 0:
        evidence = "component_source_scan_reaches_eos_top5"
    elif overall["source_margin_nonnegative"] > 0:
        evidence = "component_source_scan_crosses_eos_margin"
    elif overall["band16_strong_source_candidate"] > 0:
        evidence = "strong_blocker_band_source_candidates_found"
    elif overall["band16_source_candidate"] > 0:
        evidence = "weak_blocker_band_source_candidates_found"
    elif overall["route_eos_top50"] > 0:
        evidence = "route_near_but_no_component_source_found"
    else:
        evidence = "no_route_near_for_source_localization"
    return {
        "phase": PHASE,
        "title": "Finite Blocker Band Source Localization",
        "model": model_name,
        "status": "complete",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "attn_implementation": attn_impl,
        "selected_answer_drift_rows": selected_count,
        "overall": overall,
        "source_summaries": source_summaries,
        "bucket_summaries": bucket_summaries,
        "evidence_label": evidence,
        "boundary": (
            "Phase912 fixes the prompt-preserving route and scans layer/component suppressions to localize "
            "the finite blocker band source. Component suppression is causal source localization, not closure by itself."
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
            "selected_rows": selected_rows,
        }
        p846.write_json(out_dir / f"phase912_{args.model}_summary.json", payload)
        p846.write_jsonl(out_dir / f"phase912_{args.model}_rows.jsonl", [])
        print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
        return payload

    factors = factors_from_arg(args.factors)
    case_map = {str(case.get("case_id")): case for case in p885.extended_cases()}
    model = None
    tokenizer = None
    rows: list[dict[str, Any]] = []
    attn_impl = None
    try:
        model, tokenizer, device, attn_impl = p862.p844.p828.p796.load_model_bf16_prefer_flash(
            args.model, args.attn_implementations
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        n_layers = len(get_layers(model))
        all_specs = source_specs(model, int(args.layer_stride), factors)
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
            baseline_metrics = p903.state_metrics(tokenizer, baseline_logits, groups)
            prompt_zero_handles = p909.install_attention_input_span_scale(model, 0, 0, len(prompt_ids), 0.0)
            _prompt_zero_logits, prompt_zero_vec = p910.logits_and_l0_vector(model, device, period_ids, prompt_zero_handles)
            if base_vec is None or prompt_zero_vec is None:
                continue
            route_delta = prompt_zero_vec - base_vec
            route_delta_norm = float(torch.linalg.vector_norm(route_delta).item())
            if route_delta_norm <= 0:
                continue
            route_logits = p911.logits_with_l0_vector(model, device, period_ids, route_delta)
            if route_logits is None:
                continue
            route_metrics = p903.state_metrics(tokenizer, route_logits, groups)
            route_top_rows = p910.topk_tokens(tokenizer, route_logits, groups, max(64, int(args.band_size)))
            band32_ids = p911.top_non_eos_ids(route_top_rows, int(args.band_size))
            band16_ids = band32_ids[: min(16, len(band32_ids))]
            for spec in all_specs:
                if spec["control_kind"] == "route_only":
                    patched_logits = route_logits
                else:
                    patched_logits = logits_with_route_and_source(model, device, period_ids, route_delta, spec)
                    if patched_logits is None:
                        continue
                patched_metrics = p903.state_metrics(tokenizer, patched_logits, groups)
                patched_top_rows = p910.topk_tokens(tokenizer, patched_logits, groups, 16)
                rows.append(
                    make_row(
                        tokenizer,
                        groups,
                        source_row,
                        case,
                        spec,
                        n_layers,
                        prefix_ids,
                        prefix_text,
                        baseline_metrics,
                        route_metrics,
                        patched_metrics,
                        route_logits,
                        patched_logits,
                        route_top_rows,
                        patched_top_rows,
                        band16_ids,
                        band32_ids,
                        route_delta_norm,
                    )
                )
            if idx % max(1, int(args.log_every)) == 0 or idx == len(selected_rows):
                log(f"{args.model}/{args.round_name}: row={idx}/{len(selected_rows)} rows={len(rows)} specs={len(all_specs)}")
    finally:
        if model is not None:
            p862.p844.p828.release_model(model)
        if tokenizer is not None:
            del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    payload = summarize_model(args.model, rows, len(selected_rows), attn_impl)
    p846.write_json(out_dir / f"phase912_{args.model}_summary.json", payload)
    p846.write_jsonl(out_dir / f"phase912_{args.model}_rows.jsonl", rows)
    print(json.dumps({"phase": PHASE, "model": args.model, "overall": payload["overall"], "evidence_label": payload["evidence_label"]}, ensure_ascii=False, indent=2), flush=True)
    return payload


def summarize_round(round_name: str) -> dict[str, Any]:
    out_dir = RESULT_ROOT / round_name
    summaries = []
    scalar = Counter()
    evidence = Counter()
    top_sources = []
    top_buckets = []
    for model_name in MODELS:
        path = out_dir / f"phase912_{model_name}_summary.json"
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
            "band16_source_candidate",
            "band32_source_candidate",
            "band16_strong_source_candidate",
            "band32_strong_source_candidate",
        ]:
            scalar[key] += int(overall.get(key) or 0)
        for row in summary.get("source_summaries") or []:
            item = dict(row)
            item["model"] = summary.get("model")
            top_sources.append(item)
        for row in summary.get("bucket_summaries") or []:
            item = dict(row)
            item["model"] = summary.get("model")
            top_buckets.append(item)
    top_sources.sort(
        key=lambda row: (
            row.get("source_strict_clean_candidate") or 0,
            row.get("source_eos_top1") or 0,
            row.get("source_eos_top5") or 0,
            row.get("source_margin_nonnegative") or 0,
            row.get("band16_strong_source_candidate") or 0,
            row.get("band16_source_candidate") or 0,
            -(row.get("median_band16_mean_delta") or 9999),
            -(row.get("median_band32_mean_delta") or 9999),
        ),
        reverse=True,
    )
    top_buckets.sort(
        key=lambda row: (
            row.get("source_eos_top1") or 0,
            row.get("source_eos_top5") or 0,
            row.get("source_margin_nonnegative") or 0,
            row.get("band16_strong_source_candidate") or 0,
            -(row.get("median_band16_mean_delta") or 9999),
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
        "top_sources": top_sources[:120],
        "top_buckets": top_buckets[:60],
    }
    p846.write_json(out_dir / "phase912_cross_model_summary.json", payload)
    write_markdown(out_dir / "phase912_cross_model_summary.md", payload)
    return payload


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Phase 912 finite blocker band source localization",
        "",
        "## Overall",
        "",
        f"- models: {', '.join(payload.get('models') or [])}",
    ]
    for key, value in (payload.get("overall_scalar") or {}).items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Model Summaries", ""])
    lines.append(
        "| model | rows | source rows | route top10 | route top50 | source top1 | source top5 | source top10 | margin>=0 | band16 candidates | strong band16 | evidence |"
    )
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
    for summary in payload.get("model_summaries") or []:
        overall = summary.get("overall") or {}
        lines.append(
            "| {model} | {rows} | {source_rows} | {route10} | {route50} | {top1} | {top5} | {top10} | {margin} | {band16} | {strong16} | {evidence} |".format(
                model=summary.get("model"),
                rows=overall.get("rows"),
                source_rows=overall.get("source_rows"),
                route10=overall.get("route_eos_top10"),
                route50=overall.get("route_eos_top50"),
                top1=overall.get("source_eos_top1"),
                top5=overall.get("source_eos_top5"),
                top10=overall.get("source_eos_top10"),
                margin=overall.get("source_margin_nonnegative"),
                band16=overall.get("band16_source_candidate"),
                strong16=overall.get("band16_strong_source_candidate"),
                evidence=summary.get("evidence_label"),
            )
        )
    lines.extend(["", "## Top Sources", ""])
    lines.append(
        "| model | layer | bucket | component | factor | rows | top1 | top5 | top10 | margin>=0 | band16 cand | strong16 | median band16 mean delta | median band16 max delta | route blockers |"
    )
    lines.append("| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
    for row in payload.get("top_sources") or []:
        lines.append(
            "| {model} | {layer_idx} | {layer_bucket} | {component_kind} | {factor} | {rows} | {source_eos_top1} | {source_eos_top5} | {source_eos_top10} | {source_margin_nonnegative} | {band16_source_candidate} | {band16_strong_source_candidate} | {median_band16_mean_delta} | {median_band16_max_delta} | {route_blocker_tokens_top12} |".format(
                **row
            )
        )
    lines.extend(["", "## Top Buckets", ""])
    lines.append(
        "| model | bucket | component | factor | rows | top1 | top5 | top10 | band16 cand | median band16 mean delta |"
    )
    lines.append("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in payload.get("top_buckets") or []:
        lines.append(
            "| {model} | {layer_bucket} | {component_kind} | {factor} | {rows} | {source_eos_top1} | {source_eos_top5} | {source_eos_top10} | {band16_source_candidate} | {median_band16_mean_delta} |".format(
                **row
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--round-name", default="finite_blocker_band_source_localization")
    parser.add_argument("--phase899-round", default=PHASE899_ROUND)
    parser.add_argument("--max-rows-per-model", type=int, default=0)
    parser.add_argument("--max-prefix-tokens", type=int, default=5)
    parser.add_argument("--scale-up-factor", type=float, default=2.0)
    parser.add_argument("--layer-stride", type=int, default=1)
    parser.add_argument("--factors", default="0.5,0.0")
    parser.add_argument("--band-size", type=int, default=32)
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
