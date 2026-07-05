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
import phase910_prompt_preserving_termination_route_reconstruction as p910  # noqa: E402
import phase911_full_vocab_blocker_displacement_audit as p911  # noqa: E402
import phase912_finite_blocker_band_source_localization as p912  # noqa: E402
import phase913_route_preserving_blocker_band_disentanglement as p913  # noqa: E402
import phase918_l39_mlp_channel_a_blocker_suppressor_localization as p918  # noqa: E402
import phase919_frozen_l39_signed_margin_group_transfer_validation as p919  # noqa: E402
import phase920_consensus_l39_signed_margin_gear_holdout_controls as p920  # noqa: E402


PHASE = 921
MODELS = ["qwen3", "glm4", "deepseek7b"]
RESULT_ROOT = Path("tests/result/phase921_natural_gate_variable_search_for_l39_margin_gear")


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def parse_floats(raw: str) -> list[float]:
    return [float(part.strip()) for part in str(raw).split(",") if part.strip()]


def median(values: list[float | int | None]) -> float | None:
    cleaned = [float(value) for value in values if value is not None]
    return None if not cleaned else float(statistics.median(cleaned))


def mean(values: list[float | int | None]) -> float | None:
    cleaned = [float(value) for value in values if value is not None]
    return None if not cleaned else float(sum(cleaned) / len(cleaned))


def recapture_l39_activation(model, device: torch.device, state: dict[str, Any], target_layer: int) -> torch.Tensor | None:
    _logits, activation = p918.capture_boundary_logits_and_mlp_activation(
        model,
        device,
        state["period_ids"],
        state["route_delta"],
        state["boundary_spec"],
        len(state["prompt_ids"]),
        len(state["prefix_ids"]),
        state["l4_mlp_groups"],
        int(target_layer),
    )
    return activation


def group_contribution_stats(
    model,
    device: torch.device,
    layer_idx: int,
    activation: torch.Tensor | None,
    eos_id: int | None,
    blocker_id: int | None,
    group: list[int],
) -> dict[str, Any]:
    if activation is None or eos_id is None or blocker_id is None or not group:
        return {}
    down_proj = p913.mlp_down_proj(model, int(layer_idx))
    if down_proj is None:
        return {}
    valid = sorted({int(x) for x in group if 0 <= int(x) < int(activation.numel())})
    if not valid:
        return {}
    token_rows = p913.lm_head_rows(model, [int(eos_id), int(blocker_id)], device)
    if token_rows is None or token_rows.shape[0] < 2:
        return {}
    idx_cpu = torch.tensor(valid, dtype=torch.long)
    idx_weight = idx_cpu.to(device=down_proj.weight.device)
    act = activation.detach().float().cpu().index_select(0, idx_cpu).to(device=device, dtype=torch.float32)
    down_cols = down_proj.weight.index_select(1, idx_weight).detach().to(device=device, dtype=torch.float32)
    eos_proj = torch.matmul(token_rows[0:1], down_cols).squeeze(0)
    blocker_proj = torch.matmul(token_rows[1:2], down_cols).squeeze(0)
    eos_support = act * eos_proj
    blocker_support = act * blocker_proj
    margin_support = act * (eos_proj - blocker_proj)
    abs_act = torch.abs(act)
    pos_margin = margin_support[margin_support > 0]
    neg_margin = margin_support[margin_support < 0]
    return {
        "group_size": len(valid),
        "activation_abs_mean": float(abs_act.mean().item()),
        "activation_abs_median": float(torch.median(abs_act).item()),
        "activation_abs_l2": float(torch.linalg.vector_norm(abs_act).item()),
        "activation_signed_sum": float(act.sum().item()),
        "eos_support_sum": float(eos_support.sum().item()),
        "blocker_support_sum": float(blocker_support.sum().item()),
        "margin_support_sum": float(margin_support.sum().item()),
        "margin_support_mean": float(margin_support.mean().item()),
        "margin_support_pos_count": int(pos_margin.numel()),
        "margin_support_neg_count": int(neg_margin.numel()),
        "margin_support_pos_sum": float(pos_margin.sum().item()) if pos_margin.numel() else 0.0,
        "margin_support_neg_sum": float(neg_margin.sum().item()) if neg_margin.numel() else 0.0,
    }


def margin_vs_blocker(metrics: dict[str, Any], blocker: dict[str, Any] | None) -> float | None:
    return p911.eos_margin_vs_blocker(metrics, blocker)


def state_variable_row(
    model,
    device: torch.device,
    tokenizer,
    state: dict[str, Any],
    consensus_group: list[int],
    factors: list[float],
    groups: dict[str, list[int]],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    source_row = state["source_row"]
    case = state["case"]
    boundary_logits = state["boundary_logits"]
    boundary_metrics = state["boundary_metrics"]
    boundary_top = state["boundary_top_rows"]
    boundary_blocker = p910.first_non_eos_top(boundary_top)
    boundary_margin = margin_vs_blocker(boundary_metrics, boundary_blocker)
    route_metrics = state["route_metrics"]
    route_blocker = p910.first_non_eos_top(state["route_top_rows"])
    route_margin = margin_vs_blocker(route_metrics, route_blocker)
    l39_activation = recapture_l39_activation(model, device, state, int(args.target_layer))
    contribution = group_contribution_stats(
        model,
        device,
        int(args.target_layer),
        l39_activation,
        boundary_metrics.get("eos_best_id"),
        boundary_blocker.get("token_id") if boundary_blocker else None,
        consensus_group,
    )
    native_group = state["channel_groups"].get("margin_support_pos_64", [])
    protocol_best = boundary_metrics.get("protocol_best_logit")
    eos_logit = boundary_metrics.get("eos_best_logit")
    period_logit = boundary_metrics.get("period_best_logit")
    stop_logit = boundary_metrics.get("stop_best_logit")
    top_logit = boundary_metrics.get("next_top_logit")
    protocol_vs_eos = None if protocol_best is None or eos_logit is None else float(protocol_best - eos_logit)
    period_vs_eos = None if period_logit is None or eos_logit is None else float(period_logit - eos_logit)
    stop_vs_top = None if stop_logit is None or top_logit is None else float(stop_logit - top_logit)
    boundary_gap = None if boundary_margin is None else float(-boundary_margin)
    factor_rows: list[dict[str, Any]] = []
    min_margin_factor = None
    min_top1_factor = None
    min_strict_factor = None
    for factor in factors:
        patched_logits = p919.logits_with_target_boundary_and_frozen_group(
            model,
            device,
            state,
            consensus_group,
            int(args.target_layer),
            float(factor),
        )
        if patched_logits is None:
            continue
        patched_metrics = p903.state_metrics(tokenizer, patched_logits, groups)
        patched_top = p910.topk_tokens(tokenizer, patched_logits, groups, 16)
        patched_blocker = p910.first_non_eos_top(patched_top)
        patched_margin = margin_vs_blocker(patched_metrics, patched_blocker)
        patched_rank = patched_metrics.get("eos_rank")
        eos_top1 = bool(patched_rank == 1)
        strict = bool(p911.strict_clean_candidate(tokenizer, case, state["prefix_ids"], eos_top1))
        if patched_margin is not None and patched_margin >= 0 and min_margin_factor is None:
            min_margin_factor = float(factor)
        if eos_top1 and min_top1_factor is None:
            min_top1_factor = float(factor)
        if strict and min_strict_factor is None:
            min_strict_factor = float(factor)
        factor_rows.append(
            {
                "phase": PHASE,
                "row_kind": "phase921_low_factor_response_row",
                "model": source_row.get("model"),
                "state_key": state["state_key"],
                "case_id": source_row.get("case_id"),
                "eval_domain": source_row.get("eval_domain"),
                "prompt_variant": source_row.get("prompt_variant"),
                "source_subset_key": source_row.get("source_subset_key"),
                "edit_mode": source_row.get("edit_mode"),
                "factor": float(factor),
                "boundary_margin": boundary_margin,
                "patched_margin": patched_margin,
                "patched_eos_rank": patched_rank,
                "patched_eos_top1": eos_top1,
                "patched_margin_nonnegative": bool(patched_margin is not None and patched_margin >= 0),
                "strict_clean_candidate": strict,
                "patched_top8": patched_top[:8],
            }
        )
    variable_row = {
        "phase": PHASE,
        "row_kind": "phase921_gate_variable_state_row",
        "model": source_row.get("model"),
        "state_key": state["state_key"],
        "case_id": source_row.get("case_id"),
        "eval_domain": source_row.get("eval_domain"),
        "prompt_variant": source_row.get("prompt_variant"),
        "source_subset_key": source_row.get("source_subset_key"),
        "edit_mode": source_row.get("edit_mode"),
        "object": case.get("object"),
        "canonical_answer": case.get("canonical_answer"),
        "prefix_text": state["prefix_text"],
        "route_delta_norm": state["route_delta_norm"],
        "route_eos_rank": route_metrics.get("eos_rank"),
        "route_eos_margin_vs_blocker": route_margin,
        "route_blocker_token": route_blocker.get("token") if route_blocker else None,
        "boundary_eos_rank": boundary_metrics.get("eos_rank"),
        "boundary_eos_margin_vs_blocker": boundary_margin,
        "boundary_gap_to_zero": boundary_gap,
        "boundary_blocker_token": boundary_blocker.get("token") if boundary_blocker else None,
        "boundary_blocker_logit": boundary_blocker.get("logit") if boundary_blocker else None,
        "boundary_eos_logit": eos_logit,
        "boundary_protocol_best_logit": protocol_best,
        "boundary_period_logit": period_logit,
        "boundary_stop_logit": stop_logit,
        "boundary_top_logit": top_logit,
        "protocol_vs_eos": protocol_vs_eos,
        "period_vs_eos": period_vs_eos,
        "stop_vs_top": stop_vs_top,
        "l4_activation_abs_top": (state.get("l4_mlp_diag") or {}).get("activation_abs_top"),
        "l4_activation_abs_median": (state.get("l4_mlp_diag") or {}).get("activation_abs_median"),
        "target_native_margin_group_overlap": len(set(int(x) for x in consensus_group) & set(int(x) for x in native_group)),
        "target_native_margin_group_size": len(native_group),
        "consensus_group_size": len(consensus_group),
        "min_margin_factor": min_margin_factor,
        "min_top1_factor": min_top1_factor,
        "min_strict_factor": min_strict_factor,
        "low_factor_1375_margin": bool(min_margin_factor is not None and min_margin_factor <= 1.375),
        "low_factor_1375_top1": bool(min_top1_factor is not None and min_top1_factor <= 1.375),
        "low_factor_1375_strict": bool(min_strict_factor is not None and min_strict_factor <= 1.375),
        "boundary_top8": boundary_top[:8],
    }
    for key, value in contribution.items():
        variable_row[f"consensus_{key}"] = value
    variable_row["simple_gate_pressure"] = None
    if boundary_gap is not None and contribution.get("margin_support_sum") is not None:
        variable_row["simple_gate_pressure"] = float(boundary_gap - contribution["margin_support_sum"])
    if protocol_vs_eos is not None and boundary_gap is not None:
        variable_row["protocol_blocker_pressure"] = float(boundary_gap + max(0.0, protocol_vs_eos))
    else:
        variable_row["protocol_blocker_pressure"] = None
    return variable_row, factor_rows


def numeric_candidates(rows: list[dict[str, Any]]) -> list[str]:
    keys = set()
    for row in rows:
        for key, value in row.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                keys.add(key)
    skip = {"phase"}
    return sorted(keys - skip)


def best_threshold_accuracy(values: list[float], labels: list[bool]) -> dict[str, Any]:
    if not values or len(set(labels)) < 2:
        return {"accuracy": None, "direction": None, "threshold": None}
    pairs = sorted(zip(values, labels), key=lambda item: item[0])
    thresholds = []
    for idx in range(len(pairs) - 1):
        thresholds.append((pairs[idx][0] + pairs[idx + 1][0]) / 2.0)
    thresholds.extend([pairs[0][0] - 1e-6, pairs[-1][0] + 1e-6])
    best = {"accuracy": -1.0, "direction": None, "threshold": None}
    for threshold in thresholds:
        for direction in ["high_true", "low_true"]:
            correct = 0
            for value, label in pairs:
                pred = value >= threshold if direction == "high_true" else value <= threshold
                correct += int(bool(pred) == bool(label))
            acc = correct / len(pairs)
            if acc > best["accuracy"]:
                best = {"accuracy": float(acc), "direction": direction, "threshold": float(threshold)}
    return best


def variable_summaries(rows: list[dict[str, Any]], label_key: str = "low_factor_1375_margin") -> list[dict[str, Any]]:
    labels = [bool(row.get(label_key)) for row in rows]
    out = []
    for key in numeric_candidates(rows):
        values = []
        used_labels = []
        for row, label in zip(rows, labels):
            value = row.get(key)
            if value is None or isinstance(value, bool):
                continue
            values.append(float(value))
            used_labels.append(label)
        if len(values) < 4 or len(set(used_labels)) < 2:
            continue
        pos = [value for value, label in zip(values, used_labels) if label]
        neg = [value for value, label in zip(values, used_labels) if not label]
        threshold = best_threshold_accuracy(values, used_labels)
        out.append(
            {
                "variable": key,
                "label_key": label_key,
                "n": len(values),
                "positive_count": len(pos),
                "negative_count": len(neg),
                "positive_mean": mean(pos),
                "negative_mean": mean(neg),
                "positive_median": median(pos),
                "negative_median": median(neg),
                "mean_delta_pos_minus_neg": None if mean(pos) is None or mean(neg) is None else float(mean(pos) - mean(neg)),
                "best_threshold_accuracy": threshold.get("accuracy"),
                "best_threshold_direction": threshold.get("direction"),
                "best_threshold": threshold.get("threshold"),
            }
        )
    out.sort(
        key=lambda row: (
            row.get("best_threshold_accuracy") or 0,
            abs(row.get("mean_delta_pos_minus_neg") or 0),
        ),
        reverse=True,
    )
    return out


def summarize_factor_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[float, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[float(row.get("factor"))].append(row)
    out = []
    for factor, vals in sorted(buckets.items()):
        out.append(
            {
                "factor": factor,
                "rows": len(vals),
                "top1": sum(1 for row in vals if row.get("patched_eos_top1")),
                "margin_nonnegative": sum(1 for row in vals if row.get("patched_margin_nonnegative")),
                "strict_clean_candidate": sum(1 for row in vals if row.get("strict_clean_candidate")),
                "median_margin": median([row.get("patched_margin") for row in vals]),
            }
        )
    return out


def summarize_model(
    model_name: str,
    variable_rows: list[dict[str, Any]],
    factor_rows: list[dict[str, Any]],
    selected_count: int,
    attn_impl: str | None,
) -> dict[str, Any]:
    labels = {
        "low_factor_1375_margin": sum(1 for row in variable_rows if row.get("low_factor_1375_margin")),
        "low_factor_1375_top1": sum(1 for row in variable_rows if row.get("low_factor_1375_top1")),
        "low_factor_1375_strict": sum(1 for row in variable_rows if row.get("low_factor_1375_strict")),
    }
    if selected_count == 0:
        evidence = "no_phase915_l39_candidates"
    elif variable_rows and labels["low_factor_1375_margin"] not in {0, len(variable_rows)}:
        top_vars = variable_summaries(variable_rows, "low_factor_1375_margin")[:5]
        top_acc = max([row.get("best_threshold_accuracy") or 0 for row in top_vars] or [0])
        if top_acc >= 0.9:
            evidence = "candidate_gate_variables_separate_low_factor_closure"
        elif top_acc >= 0.75:
            evidence = "candidate_gate_variables_partially_separate_low_factor_closure"
        else:
            evidence = "no_clear_single_gate_variable"
    else:
        evidence = "insufficient_label_variation_for_gate_search"
    return {
        "phase": PHASE,
        "title": "Natural Gate Variable Search for Consensus L39 Signed Margin Gear",
        "model": model_name,
        "status": "complete",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "attn_implementation": attn_impl,
        "selected_phase915_l39_candidates": int(selected_count),
        "state_rows": len(variable_rows),
        "factor_response_rows": len(factor_rows),
        "label_counts": labels,
        "factor_summaries": summarize_factor_rows(factor_rows),
        "variable_summaries_margin1375": variable_summaries(variable_rows, "low_factor_1375_margin"),
        "variable_summaries_top1_1375": variable_summaries(variable_rows, "low_factor_1375_top1"),
        "state_summary": {
            "min_margin_factor_median": median([row.get("min_margin_factor") for row in variable_rows]),
            "min_top1_factor_median": median([row.get("min_top1_factor") for row in variable_rows]),
            "min_strict_factor_median": median([row.get("min_strict_factor") for row in variable_rows]),
            "boundary_gap_median": median([row.get("boundary_gap_to_zero") for row in variable_rows]),
            "consensus_margin_support_sum_median": median([row.get("consensus_margin_support_sum") for row in variable_rows]),
            "simple_gate_pressure_median": median([row.get("simple_gate_pressure") for row in variable_rows]),
        },
        "evidence_label": evidence,
        "boundary": (
            "Phase921 fixes the Phase920 consensus margin gear, measures low-factor closure thresholds, "
            "and ranks simple observable variables as candidate natural gate signals. It is a diagnostic "
            "gate search, not a learned gate predictor."
        ),
    }


def eval_model(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = RESULT_ROOT / args.round_name
    out_dir.mkdir(parents=True, exist_ok=True)
    selected = p918.select_phase915_candidates(args.model, args)
    if args.dry_run or not selected:
        payload = summarize_model(args.model, [], [], len(selected), None)
        payload["status"] = "dry_run" if args.dry_run else "no_phase915_l39_candidates"
        p846.write_json(out_dir / f"phase921_{args.model}_summary.json", payload)
        p846.write_jsonl(out_dir / f"phase921_{args.model}_state_rows.jsonl", [])
        p846.write_jsonl(out_dir / f"phase921_{args.model}_factor_rows.jsonl", [])
        print(json.dumps({"phase": PHASE, "model": args.model, "status": payload["status"], "selected": len(selected)}, ensure_ascii=False, indent=2), flush=True)
        return payload
    case_map = {str(case.get("case_id")): case for case in p885.extended_cases()}
    model = None
    tokenizer = None
    states: list[dict[str, Any]] = []
    variable_rows: list[dict[str, Any]] = []
    factor_rows: list[dict[str, Any]] = []
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
        factors = parse_floats(args.low_factors)
        for idx, state in enumerate(states, 1):
            row, rows = state_variable_row(model, device, tokenizer, state, consensus_group, factors, groups, args)
            row["consensus_diag"] = consensus_diag
            variable_rows.append(row)
            factor_rows.extend(rows)
            if idx % max(1, int(args.log_every)) == 0 or idx == len(states):
                log(f"{args.model}/{args.round_name}: gate_state={idx}/{len(states)} factor_rows={len(factor_rows)}")
    finally:
        if model is not None:
            p862.p844.p828.release_model(model)
        if tokenizer is not None:
            del tokenizer
        del states
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    payload = summarize_model(args.model, variable_rows, factor_rows, len(selected), attn_impl)
    p846.write_json(out_dir / f"phase921_{args.model}_summary.json", payload)
    p846.write_jsonl(out_dir / f"phase921_{args.model}_state_rows.jsonl", variable_rows)
    p846.write_jsonl(out_dir / f"phase921_{args.model}_factor_rows.jsonl", factor_rows)
    print(json.dumps({"phase": PHASE, "model": args.model, "label_counts": payload["label_counts"], "evidence_label": payload["evidence_label"]}, ensure_ascii=False, indent=2), flush=True)
    return payload


def summarize_round(round_name: str) -> dict[str, Any]:
    out_dir = RESULT_ROOT / round_name
    summaries = []
    evidence = Counter()
    scalar = Counter()
    top_variables = []
    factor_summaries = []
    for model_name in MODELS:
        path = out_dir / f"phase921_{model_name}_summary.json"
        if not path.exists():
            continue
        summary = read_json(path)
        summaries.append(summary)
        evidence[str(summary.get("evidence_label"))] += 1
        scalar["selected_phase915_l39_candidates"] += int(summary.get("selected_phase915_l39_candidates") or 0)
        scalar["state_rows"] += int(summary.get("state_rows") or 0)
        scalar["factor_response_rows"] += int(summary.get("factor_response_rows") or 0)
        for key, value in (summary.get("label_counts") or {}).items():
            scalar[key] += int(value or 0)
        for row in (summary.get("variable_summaries_margin1375") or [])[:20]:
            item = dict(row)
            item["model"] = summary.get("model")
            top_variables.append(item)
        for row in summary.get("factor_summaries") or []:
            item = dict(row)
            item["model"] = summary.get("model")
            factor_summaries.append(item)
    top_variables.sort(
        key=lambda row: (
            row.get("best_threshold_accuracy") or 0,
            abs(row.get("mean_delta_pos_minus_neg") or 0),
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
        "top_variables": top_variables[:80],
        "factor_summaries": factor_summaries,
    }
    p846.write_json(out_dir / "phase921_cross_model_summary.json", payload)
    write_markdown(out_dir / "phase921_cross_model_summary.md", payload)
    return payload


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Phase 921 natural gate variable search for consensus L39 signed margin gear",
        "",
        "## Overall",
        "",
        f"- models: {', '.join(payload.get('models') or [])}",
    ]
    for key, value in (payload.get("overall_scalar") or {}).items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Model Summaries", ""])
    lines.append("| model | selected | states | factor rows | low<=1.375 margin | low<=1.375 top1 | low<=1.375 strict | evidence |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
    for summary in payload.get("model_summaries") or []:
        labels = summary.get("label_counts") or {}
        lines.append(
            "| {model} | {selected} | {states} | {frows} | {lm} | {lt} | {ls} | {evidence} |".format(
                model=summary.get("model"),
                selected=summary.get("selected_phase915_l39_candidates"),
                states=summary.get("state_rows"),
                frows=summary.get("factor_response_rows"),
                lm=labels.get("low_factor_1375_margin"),
                lt=labels.get("low_factor_1375_top1"),
                ls=labels.get("low_factor_1375_strict"),
                evidence=summary.get("evidence_label"),
            )
        )
    lines.extend(["", "## Factor Response", ""])
    lines.append("| model | factor | rows | top1 | margin>=0 | strict | median margin |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in payload.get("factor_summaries") or []:
        lines.append(
            "| {model} | {factor} | {rows} | {top1} | {margin_nonnegative} | {strict_clean_candidate} | {median_margin} |".format(
                **row
            )
        )
    lines.extend(["", "## Top Gate Variable Candidates", ""])
    lines.append("| model | variable | n | pos | neg | pos mean | neg mean | delta | best acc | direction | threshold |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |")
    for row in payload.get("top_variables") or []:
        lines.append(
            "| {model} | {variable} | {n} | {positive_count} | {negative_count} | {positive_mean} | {negative_mean} | {mean_delta_pos_minus_neg} | {best_threshold_accuracy} | {best_threshold_direction} | {best_threshold} |".format(
                **row
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--round-name", default="natural_gate_variable_search_for_l39_margin_gear")
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
    parser.add_argument("--low-factors", default="1.125,1.25,1.375,1.5,1.75,2.0")
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
