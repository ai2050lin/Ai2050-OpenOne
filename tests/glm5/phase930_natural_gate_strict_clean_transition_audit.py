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
import phase911_full_vocab_blocker_displacement_audit as p911  # noqa: E402
import phase926_generalized_route_protocol_surface_validation as p926  # noqa: E402
import phase928_punctuation_specific_protocol_gear_search as p928  # noqa: E402
import phase929_punctuation_margin_gear_holdout_validation as p929  # noqa: E402


PHASE = 930
MODELS = ["qwen3", "glm4", "deepseek7b"]
RESULT_ROOT = Path("tests/result/phase930_natural_gate_strict_clean_transition_audit")


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def numeric(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def median(values: list[float | int | None]) -> float | None:
    cleaned = [float(value) for value in values if value is not None]
    return None if not cleaned else float(statistics.median(cleaned))


def mean(values: list[float | int | None]) -> float | None:
    cleaned = [float(value) for value in values if value is not None]
    return None if not cleaned else float(sum(cleaned) / len(cleaned))


def build_specs(args: argparse.Namespace) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = [
        {
            "control_label": "coordinate_only",
            "control_family": "coordinate_baseline",
            "control_class": "coordinate_baseline",
            "candidate_group_kind": "coordinate_only",
            "candidate_group_factor": 1.0,
        }
    ]
    for factor in p928.parse_floats(args.margin_factors):
        specs.append(
            {
                "control_label": f"L{args.target_layer}_margin_support_pos_64_gate_curve_{factor:g}",
                "control_family": "punctuation_margin_gate_curve",
                "control_class": "candidate_gear",
                "candidate_group_kind": "margin_support_pos_64",
                "candidate_group_factor": float(factor),
            }
        )
    return specs


def top_category_logit(rows: list[dict[str, Any]], categories: set[str]) -> float | None:
    vals = [numeric(row.get("logit")) for row in rows if str(row.get("category")) in categories]
    vals = [val for val in vals if val is not None]
    return None if not vals else max(vals)


def state_feature_row(tokenizer, state: dict[str, Any]) -> dict[str, Any]:
    source = state["source_row"]
    diag = state.get("channel_diag") or {}
    boundary_metrics = state.get("boundary_metrics") or {}
    boundary_top = state.get("boundary_top_rows") or []
    blocker = state.get("boundary_blocker") or {}
    eos_logit = numeric(boundary_metrics.get("eos_best_logit"))
    period_logit = top_category_logit(boundary_top, {"period"})
    punctuation_logit = top_category_logit(boundary_top, {"period", "comma", "newline"})
    protocol_logit = top_category_logit(boundary_top, {"period", "comma", "newline", "field_word", "explanation"})
    group = [int(x) for x in (state.get("channel_groups") or {}).get("margin_support_pos_64", [])]
    eos_id = getattr(tokenizer, "eos_token_id", None)
    prefix_eos_text = ""
    clean_flags: dict[str, Any] = {}
    if eos_id is not None:
        prefix_eos_text = tokenizer.decode([int(x) for x in state["prefix_ids"]] + [int(eos_id)], skip_special_tokens=True)
        clean_flags = p911.p906.clean_flags(prefix_eos_text, state["case"])
    return {
        "phase": PHASE,
        "model": source.get("model"),
        "target_state_key": state["state_key"],
        "target_case_id": source.get("case_id"),
        "target_eval_domain": source.get("eval_domain"),
        "target_prompt_variant": source.get("prompt_variant"),
        "target_object": state["case"].get("object"),
        "target_canonical_answer": state["case"].get("canonical_answer"),
        "target_prefix_text": state.get("prefix_text"),
        "prefix_eos_text": prefix_eos_text,
        "prefix_eos_text_preview": prefix_eos_text[:160],
        "prefix_eos_rollout_answer_class": bool(clean_flags.get("rollout_answer_class")),
        "prefix_eos_rollout_object_echo": bool(clean_flags.get("rollout_object_echo")),
        "prefix_eos_protocol_drift": bool(clean_flags.get("protocol_drift")),
        "prefix_eos_strict_protocol_drift": bool(clean_flags.get("strict_protocol_drift")),
        "prefix_eos_strict_clean_answer_no_protocol": bool(clean_flags.get("strict_clean_answer_no_protocol")),
        "phase925_group_kind": source.get("group_kind"),
        "phase925_factor": source.get("factor"),
        "phase925_seed_blocker_token": source.get("patched_blocker_token"),
        "phase925_seed_blocker_class": p926.blocker_class(source.get("patched_blocker_token")),
        "target_route_delta_norm": float(state.get("route_delta_norm") or 0.0),
        "target_boundary_eos_rank": boundary_metrics.get("eos_rank"),
        "target_boundary_eos_logit": eos_logit,
        "target_boundary_blocker_token": blocker.get("token"),
        "target_boundary_blocker_logit": blocker.get("logit"),
        "target_boundary_eos_margin_vs_blocker": p911.eos_margin_vs_blocker(boundary_metrics, blocker),
        "boundary_period_logit": period_logit,
        "boundary_punctuation_logit": punctuation_logit,
        "boundary_protocol_logit": protocol_logit,
        "boundary_period_gap_vs_eos": None if period_logit is None or eos_logit is None else float(period_logit - eos_logit),
        "boundary_punctuation_gap_vs_eos": None if punctuation_logit is None or eos_logit is None else float(punctuation_logit - eos_logit),
        "boundary_protocol_gap_vs_eos": None if protocol_logit is None or eos_logit is None else float(protocol_logit - eos_logit),
        "boundary_period_count_top16": sum(1 for row in boundary_top[:16] if row.get("category") == "period"),
        "boundary_punctuation_count_top16": sum(
            1 for row in boundary_top[:16] if str(row.get("category")) in {"period", "comma", "newline"}
        ),
        "l39_activation_abs_top": diag.get("activation_abs_top"),
        "l39_activation_abs_median": diag.get("activation_abs_median"),
        "l39_candidate_pool_used": diag.get("candidate_pool_used"),
        "l39_margin_pos_mean_score": diag.get("margin_support_pos_64_mean_score"),
        "l39_margin_pos_max_score": diag.get("margin_support_pos_64_max_score"),
        "l39_margin_pos_min_score": diag.get("margin_support_pos_64_min_score"),
        "l39_eos_support_mean_score": diag.get("eos_support_64_mean_score"),
        "l39_neg_margin_mean_score": diag.get("margin_support_neg_64_mean_score"),
        "margin_support_pos_64_size": len(group),
        "margin_support_pos_64_channels": group,
    }


def add_state_features_to_row(row: dict[str, Any], features: dict[str, Any], tokenizer) -> None:
    for key, value in features.items():
        if key == "margin_support_pos_64_channels":
            continue
        if key.startswith("prefix_eos_") or key.startswith("boundary_") or key.startswith("l39_"):
            row[key] = value
        elif key in {
            "phase925_group_kind",
            "phase925_factor",
            "phase925_seed_blocker_token",
            "phase925_seed_blocker_class",
        }:
            row[key] = value


def best_binary_split(rows: list[dict[str, Any]], feature: str, target: str) -> dict[str, Any] | None:
    pairs = []
    for row in rows:
        value = numeric(row.get(feature))
        if value is None or row.get(target) is None:
            continue
        pairs.append((value, bool(row.get(target))))
    if len(pairs) < 4:
        return None
    values = sorted(set(value for value, _target in pairs))
    if len(values) < 2:
        return None
    cuts = [(values[i] + values[i + 1]) / 2.0 for i in range(len(values) - 1)]
    best: dict[str, Any] | None = None
    for cut in cuts:
        for polarity in ["le_true", "ge_true"]:
            correct = 0
            for value, target_value in pairs:
                pred = value <= cut if polarity == "le_true" else value >= cut
                correct += int(pred == target_value)
            acc = correct / len(pairs)
            item = {
                "feature": feature,
                "target": target,
                "threshold": float(cut),
                "polarity": polarity,
                "correct": int(correct),
                "total": int(len(pairs)),
                "accuracy": float(acc),
                "true_count": int(sum(int(t) for _v, t in pairs)),
                "false_count": int(sum(int(not t) for _v, t in pairs)),
            }
            if best is None or item["accuracy"] > best["accuracy"]:
                best = item
    return best


def threshold_rows(rows: list[dict[str, Any]], state_features: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    features_by_state = {str(row.get("target_state_key")): row for row in state_features}
    margin_rows = [row for row in rows if row.get("candidate_group_kind") == "margin_support_pos_64"]
    by_state: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in margin_rows:
        by_state[str(row.get("target_state_key"))].append(row)
    out = []
    for state_key, srows in by_state.items():
        opens = [
            row
            for row in srows
            if row.get("new_top1_vs_coordinate_base") or row.get("new_margin_closure_vs_coordinate_base")
        ]
        best_open = min(opens, key=lambda row: (float(row.get("candidate_group_factor") or 999), -float(row.get("patched_eos_margin_vs_blocker") or -999))) if opens else None
        threshold = float(best_open.get("candidate_group_factor")) if best_open else None
        item = {
            "phase": PHASE,
            "target_state_key": state_key,
            "target_case_id": (srows[0].get("target_case_id") if srows else None),
            "opening_threshold_factor": threshold,
            "opened_at_or_below_2_00": bool(threshold is not None and threshold <= 2.00 + 1e-9),
            "opened_at_or_below_2_10": bool(threshold is not None and threshold <= 2.10 + 1e-9),
            "opened_at_or_below_2_25": bool(threshold is not None and threshold <= 2.25 + 1e-9),
            "best_opening_route_alpha": best_open.get("route_alpha") if best_open else None,
            "best_opening_protocol_factor": best_open.get("protocol_span_factor") if best_open else None,
            "best_opening_margin": best_open.get("patched_eos_margin_vs_blocker") if best_open else None,
            "best_opening_blocker_token": best_open.get("patched_blocker_token") if best_open else None,
            "strict_clean_at_opening": bool(best_open.get("strict_clean_candidate")) if best_open else False,
            "max_margin": max((float(row.get("patched_eos_margin_vs_blocker") or -9999.0) for row in srows), default=None),
            "max_margin_factor": None,
            "state_row_count": len(srows),
        }
        if srows:
            max_row = max(srows, key=lambda row: float(row.get("patched_eos_margin_vs_blocker") or -9999.0))
            item["max_margin_factor"] = max_row.get("candidate_group_factor")
            item["max_margin_route_alpha"] = max_row.get("route_alpha")
            item["max_margin_protocol_factor"] = max_row.get("protocol_span_factor")
        feat = features_by_state.get(state_key, {})
        for key, value in feat.items():
            if key != "margin_support_pos_64_channels":
                item[key] = value
        out.append(item)
    out.sort(key=lambda row: (str(row.get("target_case_id")), str(row.get("target_state_key"))))
    return out


def channel_stability(state_features: list[dict[str, Any]]) -> dict[str, Any]:
    groups = [[int(x) for x in row.get("margin_support_pos_64_channels") or []] for row in state_features]
    groups = [group for group in groups if group]
    if not groups:
        return {"state_count": 0, "union_size": 0, "intersection_size": 0, "top_channels": []}
    counter: Counter[int] = Counter()
    for group in groups:
        counter.update(set(group))
    intersection = set(groups[0])
    union = set()
    for group in groups:
        intersection &= set(group)
        union |= set(group)
    state_count = len(groups)
    return {
        "state_count": state_count,
        "group_size_median": median([len(group) for group in groups]),
        "union_size": len(union),
        "intersection_size": len(intersection),
        "channels_in_at_least_half_states": sum(1 for _ch, count in counter.items() if count >= state_count / 2),
        "channels_in_at_least_quarter_states": sum(1 for _ch, count in counter.items() if count >= state_count / 4),
        "top_channels": [{"channel": int(ch), "count": int(count)} for ch, count in counter.most_common(30)],
    }


def summarize_thresholds(trows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "states": len(trows),
        "opened": sum(1 for row in trows if row.get("opening_threshold_factor") is not None),
        "opened_at_or_below_2_00": sum(1 for row in trows if row.get("opened_at_or_below_2_00")),
        "opened_at_or_below_2_10": sum(1 for row in trows if row.get("opened_at_or_below_2_10")),
        "opened_at_or_below_2_25": sum(1 for row in trows if row.get("opened_at_or_below_2_25")),
        "strict_clean_at_opening": sum(1 for row in trows if row.get("strict_clean_at_opening")),
        "threshold_median": median([row.get("opening_threshold_factor") for row in trows]),
        "threshold_mean": mean([row.get("opening_threshold_factor") for row in trows]),
        "by_case": [
            {
                "target_case_id": case,
                "states": len(rows),
                "opened": sum(1 for row in rows if row.get("opening_threshold_factor") is not None),
                "threshold_median": median([row.get("opening_threshold_factor") for row in rows]),
                "opened_at_or_below_2_00": sum(1 for row in rows if row.get("opened_at_or_below_2_00")),
                "opened_at_or_below_2_10": sum(1 for row in rows if row.get("opened_at_or_below_2_10")),
                "opened_at_or_below_2_25": sum(1 for row in rows if row.get("opened_at_or_below_2_25")),
            }
            for case, rows in sorted(group_by(trows, "target_case_id").items())
        ],
    }


def group_by(rows: list[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(key))].append(row)
    return grouped


def gate_split_summary(trows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    features = p928.parse_csv(args.gate_features)
    splits = []
    for target in ["opened_at_or_below_2_00", "opened_at_or_below_2_10"]:
        for feature in features:
            split = best_binary_split(trows, feature, target)
            if split:
                splits.append(split)
    splits.sort(key=lambda row: (row["accuracy"], row["correct"]), reverse=True)
    return splits[:80]


def evidence_label(selected_count: int, rows: list[dict[str, Any]], trows: list[dict[str, Any]], splits: list[dict[str, Any]]) -> str:
    if selected_count <= 0:
        return "no_punctuation_period_seeds"
    if any(row.get("strict_clean_candidate") for row in rows):
        return "strict_clean_transition_found"
    if any(split.get("accuracy", 0.0) >= 0.85 for split in splits):
        return "threshold_gate_candidate_found_without_strict_clean"
    if any(row.get("opening_threshold_factor") is not None for row in trows):
        return "fine_thresholds_found_without_strict_clean"
    return "no_opening_threshold_found"


def eval_model(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = RESULT_ROOT / args.round_name
    out_dir.mkdir(parents=True, exist_ok=True)
    selected = p929.select_punctuation_seeds(args.model, args)
    coords = p928.parse_coordinate_pairs(args.coordinate_pairs)
    specs = build_specs(args)
    if args.dry_run or not selected:
        payload = {
            "phase": PHASE,
            "title": "Natural Gate and Strict-Clean Transition Audit",
            "model": args.model,
            "status": "dry_run" if args.dry_run else "no_punctuation_period_seeds",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "selected_punctuation_seeds": len(selected),
            "coordinate_count": len(coords),
            "channel_spec_count": len(specs),
            "expected_rows_if_all_reconstructed": len(selected) * len(coords) * len(specs),
            "overall": p928.summarize_rows([]),
            "threshold_summary": summarize_thresholds([]),
            "channel_stability": channel_stability([]),
            "gate_candidate_splits": [],
            "evidence_label": "no_punctuation_period_seeds" if not selected else "dry_run",
        }
        p846.write_json(out_dir / f"phase930_{args.model}_summary.json", payload)
        p846.write_jsonl(out_dir / f"phase930_{args.model}_rows.jsonl", [])
        p846.write_jsonl(out_dir / f"phase930_{args.model}_state_features.jsonl", [])
        p846.write_jsonl(out_dir / f"phase930_{args.model}_thresholds.jsonl", [])
        p846.write_jsonl(out_dir / f"phase930_{args.model}_selected_seeds.jsonl", selected)
        print(json.dumps({"phase": PHASE, "model": args.model, "status": payload["status"], "selected": len(selected)}, ensure_ascii=False, indent=2), flush=True)
        return payload

    case_map = {str(case.get("case_id")): case for case in p885.extended_cases()}
    rows: list[dict[str, Any]] = []
    state_features: list[dict[str, Any]] = []
    states: list[dict[str, Any]] = []
    model = None
    tokenizer = None
    attn_impl = None
    try:
        model, tokenizer, device, attn_impl = p862.p844.p828.p796.load_model_bf16_prefer_flash(
            args.model, args.attn_implementations
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        groups = p903.protocol_category_groups(tokenizer)
        for idx, seed in enumerate(selected, 1):
            state = p926.reconstruct_seed_state(model, tokenizer, device, groups, case_map, seed, args)
            if state is not None:
                states.append(state)
                state_features.append(state_feature_row(tokenizer, state))
            log(f"{args.model}/{args.round_name}: reconstructed_gate_seed={idx}/{len(selected)} kept={len(states)}")
        features_by_state = {str(row.get("target_state_key")): row for row in state_features}
        for state_idx, state in enumerate(states, 1):
            state_key = str(state["source_row"].get("surface_state_key"))
            features = features_by_state.get(state_key, {})
            for alpha, protocol_factor in coords:
                for base_spec in specs:
                    spec = p928.surface_spec(base_spec, alpha, protocol_factor, args.protocol_span_kind)
                    group_kind = str(spec.get("candidate_group_kind"))
                    candidate_group = [] if group_kind == "coordinate_only" else state["channel_groups"].get(group_kind, [])
                    if group_kind != "coordinate_only" and not candidate_group:
                        continue
                    patched_logits = p928.logits_with_punctuation_candidate(
                        model,
                        device,
                        state,
                        candidate_group,
                        spec,
                        int(args.target_layer),
                    )
                    if patched_logits is None:
                        continue
                    row = p928.p922.row_from_logits(
                        tokenizer,
                        state,
                        candidate_group,
                        float(spec.get("candidate_group_factor") or 1.0),
                        spec,
                        patched_logits,
                        groups,
                    )
                    row["phase"] = PHASE
                    row["row_kind"] = "phase930_natural_gate_strict_clean_row"
                    row["phase925_surface_state_key"] = state_key
                    row["candidate_group_kind"] = group_kind
                    row["candidate_group_factor"] = float(spec.get("candidate_group_factor") or 1.0)
                    row["candidate_group_size"] = len(candidate_group)
                    add_state_features_to_row(row, features, tokenizer)
                    rows.append(row)
            if state_idx % max(1, int(args.log_every)) == 0 or state_idx == len(states):
                log(f"{args.model}/{args.round_name}: gate_state={state_idx}/{len(states)} rows={len(rows)}")
    finally:
        if model is not None:
            p862.p844.p828.release_model(model)
        if tokenizer is not None:
            del tokenizer
        del states
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    p928.annotate_vs_coordinate_baseline(rows)
    trows = threshold_rows(rows, state_features, args)
    splits = gate_split_summary(trows, args)
    label = evidence_label(len(selected), rows, trows, splits)
    payload = {
        "phase": PHASE,
        "title": "Natural Gate and Strict-Clean Transition Audit",
        "model": args.model,
        "status": "complete",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "attn_implementation": attn_impl,
        "selected_punctuation_seeds": len(selected),
        "coordinate_count": len(coords),
        "channel_spec_count": len(specs),
        "expected_rows_if_all_reconstructed": len(selected) * len(coords) * len(specs),
        "overall": p928.summarize_rows(rows),
        "threshold_summary": summarize_thresholds(trows),
        "channel_stability": channel_stability(state_features),
        "by_factor": p928.summarize_by(rows, ["candidate_group_kind", "candidate_group_factor"], limit=120),
        "by_case_factor": p928.summarize_by(rows, ["target_case_id", "candidate_group_factor"], limit=240),
        "gate_candidate_splits": splits,
        "evidence_label": label,
        "boundary": "fine threshold and gate-candidate audit only; no natural gate closure claim",
    }
    p846.write_json(out_dir / f"phase930_{args.model}_summary.json", payload)
    p846.write_jsonl(out_dir / f"phase930_{args.model}_rows.jsonl", rows)
    p846.write_jsonl(out_dir / f"phase930_{args.model}_state_features.jsonl", state_features)
    p846.write_jsonl(out_dir / f"phase930_{args.model}_thresholds.jsonl", trows)
    p846.write_jsonl(out_dir / f"phase930_{args.model}_selected_seeds.jsonl", selected)
    print(json.dumps({"phase": PHASE, "model": args.model, "status": label, "overall": payload["overall"], "thresholds": payload["threshold_summary"]}, ensure_ascii=False, indent=2), flush=True)
    return payload


def summarize_round(round_name: str) -> dict[str, Any]:
    out_dir = RESULT_ROOT / round_name
    summaries = [read_json(out_dir / f"phase930_{model}_summary.json") for model in MODELS]
    summaries = [summary for summary in summaries if summary]
    evidence_counts: Counter[str] = Counter()
    overall_scalar: dict[str, Any] = {}
    all_splits: list[dict[str, Any]] = []
    for summary in summaries:
        evidence_counts[str(summary.get("evidence_label"))] += 1
        overall_scalar["selected_punctuation_seeds"] = overall_scalar.get("selected_punctuation_seeds", 0) + int(summary.get("selected_punctuation_seeds") or 0)
        overall_scalar["expected_rows_if_all_reconstructed"] = overall_scalar.get("expected_rows_if_all_reconstructed", 0) + int(summary.get("expected_rows_if_all_reconstructed") or 0)
        for key, value in (summary.get("overall") or {}).items():
            if isinstance(value, int):
                overall_scalar[f"overall_{key}"] = overall_scalar.get(f"overall_{key}", 0) + value
        tsum = summary.get("threshold_summary") or {}
        for key in ["states", "opened", "opened_at_or_below_2_00", "opened_at_or_below_2_10", "opened_at_or_below_2_25", "strict_clean_at_opening"]:
            overall_scalar[f"threshold_{key}"] = overall_scalar.get(f"threshold_{key}", 0) + int(tsum.get(key) or 0)
        for split in summary.get("gate_candidate_splits") or []:
            item = dict(split)
            item["model"] = summary.get("model")
            all_splits.append(item)
    all_splits.sort(key=lambda row: (float(row.get("accuracy") or 0.0), int(row.get("correct") or 0)), reverse=True)
    payload = {
        "phase": PHASE,
        "round": round_name,
        "status": "complete" if summaries else "missing",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "models": [summary.get("model") for summary in summaries],
        "overall_scalar": overall_scalar,
        "evidence_label_counts": dict(evidence_counts),
        "top_gate_candidate_splits": all_splits[:80],
        "model_summaries": summaries,
    }
    p846.write_json(out_dir / "phase930_cross_model_summary.json", payload)
    write_summary_md(out_dir / "phase930_cross_model_summary.md", payload)
    return payload


def write_summary_md(path: Path, payload: dict[str, Any]) -> None:
    lines = ["# Phase 930 natural gate and strict-clean transition audit", "", "## Overall", ""]
    for key, value in sorted((payload.get("overall_scalar") or {}).items()):
        lines.append(f"- {key}: {value}")
    lines += ["", "## Evidence", ""]
    for key, value in sorted((payload.get("evidence_label_counts") or {}).items()):
        lines.append(f"- {key}: {value}")
    lines += ["", "## Top Gate Candidate Splits", ""]
    lines.append("| model | feature | target | threshold | polarity | accuracy | correct | total | true | false |")
    lines.append("| --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |")
    for row in payload.get("top_gate_candidate_splits") or []:
        lines.append(
            "| {model} | {feature} | {target} | {threshold} | {polarity} | {accuracy} | {correct} | {total} | {true_count} | {false_count} |".format(
                **row
            )
        )
    lines += ["", "## Model Thresholds", ""]
    for summary in payload.get("model_summaries") or []:
        if not summary.get("threshold_summary"):
            continue
        lines.append(f"### {summary.get('model')}")
        lines.append("")
        for key, value in (summary.get("threshold_summary") or {}).items():
            if key != "by_case":
                lines.append(f"- {key}: {value}")
        lines.append("- by_case:")
        for row in (summary.get("threshold_summary") or {}).get("by_case") or []:
            lines.append(
                "  - {target_case_id}: states={states}, opened={opened}, threshold_median={threshold_median}, <=2.00={opened_at_or_below_2_00}, <=2.10={opened_at_or_below_2_10}, <=2.25={opened_at_or_below_2_25}".format(
                    **row
                )
            )
        stability = summary.get("channel_stability") or {}
        lines.append(f"- channel_stability: union={stability.get('union_size')}, intersection={stability.get('intersection_size')}, half={stability.get('channels_in_at_least_half_states')}, quarter={stability.get('channels_in_at_least_quarter_states')}")
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--round-name", default="natural_gate_strict_clean_transition_audit")
    parser.add_argument("--phase925-round", default="response_surface_generalization_dataset_expansion")
    parser.add_argument("--seed-source", choices=["selected", "candidate"], default="selected")
    parser.add_argument("--max-punctuation-seeds", type=int, default=30)
    parser.add_argument("--max-per-case", type=int, default=10)
    parser.add_argument("--coordinate-pairs", default="1.0:1.0,0.875:1.1,1.25:1.1,0.875:0.85,1.25:0.85,1.375:0.85,1.375:0.9")
    parser.add_argument("--margin-factors", default="2.0,2.05,2.1,2.15,2.2,2.25")
    parser.add_argument("--protocol-span-kind", default="last8_before_period")
    parser.add_argument("--target-layer", type=int, default=39)
    parser.add_argument("--max-prefix-tokens", type=int, default=5)
    parser.add_argument("--scale-up-factor", type=float, default=2.0)
    parser.add_argument("--l4-candidate-pool", type=int, default=512)
    parser.add_argument("--channel-candidate-pool", type=int, default=768)
    parser.add_argument("--band-size", type=int, default=32)
    parser.add_argument("--gate-features", default="target_route_delta_norm,target_boundary_eos_margin_vs_blocker,target_boundary_eos_rank,target_boundary_blocker_logit,boundary_period_gap_vs_eos,boundary_punctuation_gap_vs_eos,l39_activation_abs_top,l39_activation_abs_median,l39_margin_pos_mean_score,l39_margin_pos_max_score,l39_margin_pos_min_score,l39_eos_support_mean_score,l39_neg_margin_mean_score,phase925_factor")
    parser.add_argument("--log-every", type=int, default=5)
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
