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
import phase903_protocol_continuation_field_mapping as p903  # noqa: E402
import phase918_l39_mlp_channel_a_blocker_suppressor_localization as p918  # noqa: E402
import phase919_frozen_l39_signed_margin_group_transfer_validation as p919  # noqa: E402
import phase920_consensus_l39_signed_margin_gear_holdout_controls as p920  # noqa: E402
import phase922_candidate_gate_variable_causal_coupling_test as p922  # noqa: E402


PHASE = 923
MODELS = ["qwen3", "glm4", "deepseek7b"]
RESULT_ROOT = Path("tests/result/phase923_route_alpha_response_curve_audit")


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


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


def route_alpha_spec(alpha: float) -> dict[str, Any]:
    return {
        "control_label": f"route_alpha_{float(alpha):g}",
        "control_family": "route_alpha_curve",
        "control_class": "route_alpha_curve",
        "route_alpha": float(alpha),
        "l4_factor_multiplier": 1.0,
    }


def annotate_vs_alpha_one(rows: list[dict[str, Any]]) -> None:
    baselines: dict[tuple[str, float], dict[str, Any]] = {}
    for row in rows:
        if abs(float(row.get("route_alpha") or 0.0) - 1.0) <= 1e-9:
            baselines[(str(row.get("target_state_key")), float(row.get("l39_factor")))] = row
    for row in rows:
        base = baselines.get((str(row.get("target_state_key")), float(row.get("l39_factor"))))
        if base is None:
            continue
        row["alpha1_margin"] = base.get("patched_eos_margin_vs_blocker")
        row["alpha1_eos_rank"] = base.get("patched_eos_rank")
        row["alpha1_top1"] = base.get("patched_eos_top1")
        row["alpha1_margin_nonnegative"] = base.get("patched_eos_margin_nonnegative")
        row["alpha1_strict_clean_candidate"] = base.get("strict_clean_candidate")
        row_margin = row.get("patched_eos_margin_vs_blocker")
        base_margin = base.get("patched_eos_margin_vs_blocker")
        row_rank = row.get("patched_eos_rank")
        base_rank = base.get("patched_eos_rank")
        row["margin_delta_vs_alpha1"] = None if row_margin is None or base_margin is None else float(row_margin - base_margin)
        row["rank_delta_vs_alpha1"] = None if row_rank is None or base_rank is None else int(row_rank) - int(base_rank)
        row["improved_margin_vs_alpha1"] = bool(
            row["margin_delta_vs_alpha1"] is not None and row["margin_delta_vs_alpha1"] > 0
        )
        row["worsened_margin_vs_alpha1"] = bool(
            row["margin_delta_vs_alpha1"] is not None and row["margin_delta_vs_alpha1"] < 0
        )
        row["new_margin_closure_vs_alpha1"] = bool(
            not base.get("patched_eos_margin_nonnegative") and row.get("patched_eos_margin_nonnegative")
        )
        row["lost_margin_closure_vs_alpha1"] = bool(
            base.get("patched_eos_margin_nonnegative") and not row.get("patched_eos_margin_nonnegative")
        )
        row["new_top1_vs_alpha1"] = bool(not base.get("patched_eos_top1") and row.get("patched_eos_top1"))
        row["lost_top1_vs_alpha1"] = bool(base.get("patched_eos_top1") and not row.get("patched_eos_top1"))
        row["new_strict_vs_alpha1"] = bool(
            not base.get("strict_clean_candidate") and row.get("strict_clean_candidate")
        )
        row["lost_strict_vs_alpha1"] = bool(
            base.get("strict_clean_candidate") and not row.get("strict_clean_candidate")
        )


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "rows": len(rows),
        "top1": sum(1 for row in rows if row.get("patched_eos_top1")),
        "margin_nonnegative": sum(1 for row in rows if row.get("patched_eos_margin_nonnegative")),
        "strict_clean_candidate": sum(1 for row in rows if row.get("strict_clean_candidate")),
        "improved_margin_vs_alpha1": sum(1 for row in rows if row.get("improved_margin_vs_alpha1")),
        "worsened_margin_vs_alpha1": sum(1 for row in rows if row.get("worsened_margin_vs_alpha1")),
        "new_margin_closure_vs_alpha1": sum(1 for row in rows if row.get("new_margin_closure_vs_alpha1")),
        "lost_margin_closure_vs_alpha1": sum(1 for row in rows if row.get("lost_margin_closure_vs_alpha1")),
        "new_top1_vs_alpha1": sum(1 for row in rows if row.get("new_top1_vs_alpha1")),
        "lost_top1_vs_alpha1": sum(1 for row in rows if row.get("lost_top1_vs_alpha1")),
        "new_strict_vs_alpha1": sum(1 for row in rows if row.get("new_strict_vs_alpha1")),
        "lost_strict_vs_alpha1": sum(1 for row in rows if row.get("lost_strict_vs_alpha1")),
        "median_margin_delta_vs_alpha1": median([row.get("margin_delta_vs_alpha1") for row in rows]),
        "mean_margin_delta_vs_alpha1": mean([row.get("margin_delta_vs_alpha1") for row in rows]),
        "median_patched_margin": median([row.get("patched_eos_margin_vs_blocker") for row in rows]),
        "mean_patched_margin": mean([row.get("patched_eos_margin_vs_blocker") for row in rows]),
        "target_state_coverage_top1": len({row.get("target_state_key") for row in rows if row.get("patched_eos_top1")}),
        "target_state_coverage_margin": len(
            {row.get("target_state_key") for row in rows if row.get("patched_eos_margin_nonnegative")}
        ),
    }


def summarize_by(rows: list[dict[str, Any]], keys: list[str], limit: int = 200) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[tuple(str(row.get(key)) for key in keys)].append(row)
    out = []
    for key_tuple, vals in buckets.items():
        summary = summarize_rows(vals)
        first = vals[0]
        for name, value in zip(keys, key_tuple):
            summary[name] = value
        for meta_key in ["route_alpha", "l39_factor", "control_label", "control_family", "control_class"]:
            summary.setdefault(meta_key, first.get(meta_key))
        out.append(summary)
    out.sort(
        key=lambda row: (
            row.get("new_strict_vs_alpha1") or 0,
            row.get("new_top1_vs_alpha1") or 0,
            row.get("new_margin_closure_vs_alpha1") or 0,
            row.get("improved_margin_vs_alpha1") or 0,
            row.get("mean_margin_delta_vs_alpha1") or -9999,
        ),
        reverse=True,
    )
    return out[:limit]


def curve_summaries(rows: list[dict[str, Any]], alpha_values: list[float]) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, float], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[(str(row.get("target_state_key")), float(row.get("l39_factor")))].append(row)
    out = []
    sorted_alphas = sorted(float(a) for a in alpha_values)
    for (state_key, factor), vals in buckets.items():
        vals_by_alpha = {float(row.get("route_alpha")): row for row in vals}
        margins = [vals_by_alpha.get(alpha, {}).get("patched_eos_margin_vs_blocker") for alpha in sorted_alphas]
        numeric = [(alpha, float(margin)) for alpha, margin in zip(sorted_alphas, margins) if margin is not None]
        if not numeric:
            continue
        best_alpha, best_margin = max(numeric, key=lambda item: (item[1], -abs(item[0] - 1.0)))
        alpha1_margin = vals_by_alpha.get(1.0, {}).get("patched_eos_margin_vs_blocker")
        monotonic_non_decreasing = True
        monotonic_non_increasing = True
        last_margin = None
        for _alpha, margin in numeric:
            if last_margin is not None and margin < last_margin:
                monotonic_non_decreasing = False
            if last_margin is not None and margin > last_margin:
                monotonic_non_increasing = False
            last_margin = margin
        closure_alphas = [
            alpha
            for alpha, row in vals_by_alpha.items()
            if row.get("patched_eos_margin_nonnegative") and row.get("patched_eos_top1")
        ]
        first = vals[0]
        out.append(
            {
                "target_state_key": state_key,
                "target_case_id": first.get("target_case_id"),
                "target_eval_domain": first.get("target_eval_domain"),
                "target_object": first.get("target_object"),
                "l39_factor": float(factor),
                "best_alpha": float(best_alpha),
                "best_margin": float(best_margin),
                "alpha1_margin": alpha1_margin,
                "best_margin_delta_vs_alpha1": None
                if alpha1_margin is None
                else float(best_margin - float(alpha1_margin)),
                "best_alpha_lt_1": bool(best_alpha < 1.0),
                "best_alpha_eq_1": bool(abs(best_alpha - 1.0) <= 1e-9),
                "best_alpha_gt_1": bool(best_alpha > 1.0),
                "monotonic_non_decreasing": bool(monotonic_non_decreasing),
                "monotonic_non_increasing": bool(monotonic_non_increasing),
                "closure_alpha_count": len(closure_alphas),
                "closure_alphas": sorted(float(alpha) for alpha in closure_alphas),
            }
        )
    out.sort(
        key=lambda row: (
            row.get("closure_alpha_count") or 0,
            row.get("best_margin_delta_vs_alpha1") or -9999,
        ),
        reverse=True,
    )
    return out


def summarize_curves(curves: list[dict[str, Any]]) -> dict[str, Any]:
    best_counter = Counter(str(row.get("best_alpha")) for row in curves)
    return {
        "curve_count": len(curves),
        "best_alpha_distribution": dict(sorted(best_counter.items())),
        "best_alpha_lt_1": sum(1 for row in curves if row.get("best_alpha_lt_1")),
        "best_alpha_eq_1": sum(1 for row in curves if row.get("best_alpha_eq_1")),
        "best_alpha_gt_1": sum(1 for row in curves if row.get("best_alpha_gt_1")),
        "monotonic_non_decreasing": sum(1 for row in curves if row.get("monotonic_non_decreasing")),
        "monotonic_non_increasing": sum(1 for row in curves if row.get("monotonic_non_increasing")),
        "with_closure_alpha": sum(1 for row in curves if (row.get("closure_alpha_count") or 0) > 0),
        "median_best_margin_delta_vs_alpha1": median([row.get("best_margin_delta_vs_alpha1") for row in curves]),
        "mean_best_margin_delta_vs_alpha1": mean([row.get("best_margin_delta_vs_alpha1") for row in curves]),
    }


def summarize_model(
    model_name: str,
    rows: list[dict[str, Any]],
    curves: list[dict[str, Any]],
    selected_count: int,
    alpha_count: int,
    factor_count: int,
    consensus_diag: dict[str, Any] | None,
    attn_impl: str | None,
) -> dict[str, Any]:
    alpha1_rows = [row for row in rows if abs(float(row.get("route_alpha") or 0.0) - 1.0) <= 1e-9]
    non_alpha1_rows = [row for row in rows if abs(float(row.get("route_alpha") or 0.0) - 1.0) > 1e-9]
    overall = {
        "all": summarize_rows(rows),
        "alpha1": summarize_rows(alpha1_rows),
        "non_alpha1": summarize_rows(non_alpha1_rows),
        "target_state_count": len({row.get("target_state_key") for row in rows}),
    }
    curve_summary = summarize_curves(curves)
    if selected_count == 0:
        evidence = "no_phase915_l39_candidates"
    elif curve_summary["best_alpha_lt_1"] > 0 and curve_summary["best_alpha_gt_1"] > 0:
        evidence = "route_alpha_response_nonmonotonic_mixed_peak"
    elif curve_summary["best_alpha_lt_1"] > 0:
        evidence = "route_alpha_response_has_low_alpha_peak"
    elif curve_summary["best_alpha_gt_1"] > 0:
        evidence = "route_alpha_response_has_high_alpha_peak"
    else:
        evidence = "route_alpha_response_flat_or_alpha1_peak"
    return {
        "phase": PHASE,
        "title": "Route Alpha Response Curve Audit",
        "model": model_name,
        "status": "complete",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "attn_implementation": attn_impl,
        "selected_phase915_l39_candidates": int(selected_count),
        "route_alpha_count": int(alpha_count),
        "low_factor_count": int(factor_count),
        "consensus_diag": consensus_diag or {},
        "overall": overall,
        "curve_summary": curve_summary,
        "by_alpha": summarize_by(rows, ["route_alpha"], limit=80),
        "by_factor": summarize_by(rows, ["l39_factor"], limit=50),
        "by_alpha_factor": summarize_by(rows, ["route_alpha", "l39_factor"], limit=200),
        "top_curves": curves[:80],
        "evidence_label": evidence,
        "boundary": (
            "Phase923 scans route_alpha with fixed L39 consensus gear and low L39 factors. It tests whether "
            "the route variable behaves like a monotonic gate or a local route-gear matching curve."
        ),
    }


def eval_model(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = RESULT_ROOT / args.round_name
    out_dir.mkdir(parents=True, exist_ok=True)
    selected = p918.select_phase915_candidates(args.model, args)
    alpha_values = parse_floats(args.route_alphas)
    low_factors = parse_floats(args.low_factors)
    if args.dry_run or not selected:
        payload = summarize_model(args.model, [], [], len(selected), len(alpha_values), len(low_factors), {}, None)
        payload["status"] = "dry_run" if args.dry_run else "no_phase915_l39_candidates"
        p846.write_json(out_dir / f"phase923_{args.model}_summary.json", payload)
        p846.write_jsonl(out_dir / f"phase923_{args.model}_rows.jsonl", [])
        p846.write_jsonl(out_dir / f"phase923_{args.model}_curves.jsonl", [])
        print(
            json.dumps(
                {"phase": PHASE, "model": args.model, "status": payload["status"], "selected": len(selected)},
                ensure_ascii=False,
                indent=2,
            ),
            flush=True,
        )
        return payload
    case_map = {str(case.get("case_id")): case for case in p885.extended_cases()}
    model = None
    tokenizer = None
    states: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    consensus_diag: dict[str, Any] = {}
    attn_impl = None
    try:
        model, tokenizer, device, attn_impl = p862.p844.p828.p796.load_model_bf16_prefer_flash(
            args.model, args.attn_implementations
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        groups = p903.protocol_category_groups(tokenizer)
        for idx, source_row in enumerate(selected, 1):
            state = p919.reconstruct_state(model, tokenizer, device, groups, case_map, source_row, args)
            if state is not None:
                states.append(state)
            log(f"{args.model}/{args.round_name}: reconstructed_state={idx}/{len(selected)} kept={len(states)}")
        consensus_group, consensus_diag = p920.consensus_group(states, "margin_support_pos_64", int(args.group_budget))
        for state_idx, state in enumerate(states, 1):
            for l39_factor in low_factors:
                for alpha in alpha_values:
                    spec = route_alpha_spec(float(alpha))
                    patched_logits = p922.logits_with_coupled_intervention(
                        model,
                        device,
                        state,
                        consensus_group,
                        float(l39_factor),
                        spec,
                        int(args.target_layer),
                    )
                    if patched_logits is None:
                        continue
                    row = p922.row_from_logits(tokenizer, state, consensus_group, float(l39_factor), spec, patched_logits, groups)
                    row["phase"] = PHASE
                    row["row_kind"] = "phase923_route_alpha_response_curve_row"
                    rows.append(row)
            if state_idx % max(1, int(args.log_every)) == 0 or state_idx == len(states):
                log(f"{args.model}/{args.round_name}: route_alpha_state={state_idx}/{len(states)} rows={len(rows)}")
    finally:
        if model is not None:
            p862.p844.p828.release_model(model)
        if tokenizer is not None:
            del tokenizer
        del states
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    annotate_vs_alpha_one(rows)
    curves = curve_summaries(rows, alpha_values)
    payload = summarize_model(
        args.model,
        rows,
        curves,
        len(selected),
        len(alpha_values),
        len(low_factors),
        consensus_diag,
        attn_impl,
    )
    p846.write_json(out_dir / f"phase923_{args.model}_summary.json", payload)
    p846.write_jsonl(out_dir / f"phase923_{args.model}_rows.jsonl", rows)
    p846.write_jsonl(out_dir / f"phase923_{args.model}_curves.jsonl", curves)
    print(
        json.dumps(
            {
                "phase": PHASE,
                "model": args.model,
                "curve_summary": payload["curve_summary"],
                "evidence_label": payload["evidence_label"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return payload


def summarize_round(round_name: str) -> dict[str, Any]:
    out_dir = RESULT_ROOT / round_name
    summaries = []
    evidence = Counter()
    scalar = Counter()
    alphas = []
    alpha_factors = []
    curves = []
    for model_name in MODELS:
        path = out_dir / f"phase923_{model_name}_summary.json"
        if not path.exists():
            continue
        summary = read_json(path)
        summaries.append(summary)
        evidence[str(summary.get("evidence_label"))] += 1
        scalar["selected_phase915_l39_candidates"] += int(summary.get("selected_phase915_l39_candidates") or 0)
        overall = summary.get("overall") or {}
        for scope in ["all", "alpha1", "non_alpha1"]:
            scoped = overall.get(scope) or {}
            for key in [
                "rows",
                "top1",
                "margin_nonnegative",
                "strict_clean_candidate",
                "improved_margin_vs_alpha1",
                "new_margin_closure_vs_alpha1",
                "new_top1_vs_alpha1",
                "lost_margin_closure_vs_alpha1",
            ]:
                scalar[f"{scope}_{key}"] += int(scoped.get(key) or 0)
        csum = summary.get("curve_summary") or {}
        for key in ["curve_count", "best_alpha_lt_1", "best_alpha_eq_1", "best_alpha_gt_1", "monotonic_non_decreasing", "monotonic_non_increasing", "with_closure_alpha"]:
            scalar[f"curves_{key}"] += int(csum.get(key) or 0)
        scalar["target_state_count"] += int(overall.get("target_state_count") or 0)
        for source_key, target in [("by_alpha", alphas), ("by_alpha_factor", alpha_factors), ("top_curves", curves)]:
            for row in summary.get(source_key) or []:
                item = dict(row)
                item["model"] = summary.get("model")
                target.append(item)
    sort_rows = lambda row: (
        row.get("new_strict_vs_alpha1") or 0,
        row.get("new_top1_vs_alpha1") or 0,
        row.get("new_margin_closure_vs_alpha1") or 0,
        row.get("improved_margin_vs_alpha1") or 0,
        row.get("mean_margin_delta_vs_alpha1") or -9999,
    )
    alphas.sort(key=sort_rows, reverse=True)
    alpha_factors.sort(key=sort_rows, reverse=True)
    curves.sort(key=lambda row: (row.get("closure_alpha_count") or 0, row.get("best_margin_delta_vs_alpha1") or -9999), reverse=True)
    payload = {
        "phase": PHASE,
        "round": round_name,
        "status": "complete" if len(summaries) == len(MODELS) else "partial",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "models": [summary.get("model") for summary in summaries],
        "overall_scalar": {key: int(value) for key, value in sorted(scalar.items())},
        "evidence_label_counts": dict(sorted(evidence.items())),
        "model_summaries": summaries,
        "top_alphas": alphas[:80],
        "top_alpha_factors": alpha_factors[:160],
        "top_curves": curves[:120],
    }
    p846.write_json(out_dir / "phase923_cross_model_summary.json", payload)
    write_markdown(out_dir / "phase923_cross_model_summary.md", payload)
    return payload


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Phase 923 route alpha response curve audit",
        "",
        "## Overall",
        "",
        f"- models: {', '.join(payload.get('models') or [])}",
    ]
    for key, value in (payload.get("overall_scalar") or {}).items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Model Summaries", ""])
    lines.append(
        "| model | selected | states | curves | best<1 | best=1 | best>1 | mono up | mono down | alpha1 top1 | non-alpha1 new top1 | evidence |"
    )
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
    for summary in payload.get("model_summaries") or []:
        overall = summary.get("overall") or {}
        alpha1 = overall.get("alpha1") or {}
        non_alpha1 = overall.get("non_alpha1") or {}
        curves = summary.get("curve_summary") or {}
        lines.append(
            "| {model} | {selected} | {states} | {curves} | {lt} | {eq} | {gt} | {mono_up} | {mono_down} | {alpha1_top1} | {non_new_top1} | {evidence} |".format(
                model=summary.get("model"),
                selected=summary.get("selected_phase915_l39_candidates"),
                states=overall.get("target_state_count"),
                curves=curves.get("curve_count"),
                lt=curves.get("best_alpha_lt_1"),
                eq=curves.get("best_alpha_eq_1"),
                gt=curves.get("best_alpha_gt_1"),
                mono_up=curves.get("monotonic_non_decreasing"),
                mono_down=curves.get("monotonic_non_increasing"),
                alpha1_top1=alpha1.get("top1"),
                non_new_top1=non_alpha1.get("new_top1_vs_alpha1"),
                evidence=summary.get("evidence_label"),
            )
        )
    lines.extend(["", "## Top Alphas", ""])
    lines.append("| model | alpha | rows | top1 | margin | strict | improved | new margin | new top1 | lost margin | mean delta | median margin |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in payload.get("top_alphas") or []:
        row = {
            "model": "",
            "route_alpha": "",
            "rows": 0,
            "top1": 0,
            "margin_nonnegative": 0,
            "strict_clean_candidate": 0,
            "improved_margin_vs_alpha1": 0,
            "new_margin_closure_vs_alpha1": 0,
            "new_top1_vs_alpha1": 0,
            "lost_margin_closure_vs_alpha1": 0,
            "mean_margin_delta_vs_alpha1": None,
            "median_patched_margin": None,
            **row,
        }
        lines.append(
            "| {model} | {route_alpha} | {rows} | {top1} | {margin_nonnegative} | {strict_clean_candidate} | {improved_margin_vs_alpha1} | {new_margin_closure_vs_alpha1} | {new_top1_vs_alpha1} | {lost_margin_closure_vs_alpha1} | {mean_margin_delta_vs_alpha1} | {median_patched_margin} |".format(
                **row
            )
        )
    lines.extend(["", "## Top Curves", ""])
    lines.append("| model | state | factor | best alpha | best margin | alpha1 margin | delta | closure alphas | monotonic up | monotonic down |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |")
    for row in payload.get("top_curves") or []:
        row = {
            "model": "",
            "target_state_key": "",
            "l39_factor": "",
            "best_alpha": "",
            "best_margin": "",
            "alpha1_margin": "",
            "best_margin_delta_vs_alpha1": None,
            "closure_alphas": [],
            "monotonic_non_decreasing": False,
            "monotonic_non_increasing": False,
            **row,
        }
        lines.append(
            "| {model} | {target_state_key} | {l39_factor} | {best_alpha} | {best_margin} | {alpha1_margin} | {best_margin_delta_vs_alpha1} | {closure_alphas} | {monotonic_non_decreasing} | {monotonic_non_increasing} |".format(
                **row
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--round-name", default="route_alpha_response_curve_audit")
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
    parser.add_argument("--group-budget", type=int, default=64)
    parser.add_argument("--low-factors", default="1.125,1.25,1.375")
    parser.add_argument("--route-alphas", default="0.5,0.625,0.75,0.875,1.0,1.125,1.25,1.375,1.5")
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
