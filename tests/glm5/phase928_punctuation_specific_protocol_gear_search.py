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
import phase913_route_preserving_blocker_band_disentanglement as p913  # noqa: E402
import phase918_l39_mlp_channel_a_blocker_suppressor_localization as p918  # noqa: E402
import phase922_candidate_gate_variable_causal_coupling_test as p922  # noqa: E402
import phase926_generalized_route_protocol_surface_validation as p926  # noqa: E402


PHASE = 928
MODELS = ["qwen3", "glm4", "deepseek7b"]
RESULT_ROOT = Path("tests/result/phase928_punctuation_specific_protocol_gear_search")
PHASE926_ROOT = Path("tests/result/phase926_generalized_route_protocol_surface_validation")


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


def parse_coordinate_pairs(raw: str) -> list[tuple[float, float]]:
    pairs = []
    for part in parse_csv(raw):
        if ":" not in part:
            raise ValueError(f"coordinate pair must be alpha:protocol, got {part!r}")
        left, right = part.split(":", 1)
        pairs.append((float(left), float(right)))
    return pairs


def median(values: list[float | int | None]) -> float | None:
    cleaned = [float(value) for value in values if value is not None]
    return None if not cleaned else float(statistics.median(cleaned))


def mean(values: list[float | int | None]) -> float | None:
    cleaned = [float(value) for value in values if value is not None]
    return None if not cleaned else float(sum(cleaned) / len(cleaned))


def blocker_class(token: Any) -> str:
    text = "" if token is None else str(token)
    stripped = text.strip()
    if stripped in {".", "。"}:
        return "punctuation_period"
    if stripped == "a":
        return "article_a"
    if not stripped:
        return "blank_or_space"
    return "other"


def seed_sort_key(row: dict[str, Any]) -> tuple[int, float, str]:
    return (
        int(blocker_class(row.get("patched_blocker_token")) == "punctuation_period"),
        float(row.get("score_scalar") or 0.0),
        str(row.get("surface_state_key")),
    )


def select_punctuation_seeds(model_name: str, args: argparse.Namespace) -> list[dict[str, Any]]:
    path = PHASE926_ROOT / args.phase926_round / f"phase926_{model_name}_selected_seeds.jsonl"
    rows = [
        row
        for row in read_jsonl(path)
        if blocker_class(row.get("patched_blocker_token")) == "punctuation_period"
    ]
    rows.sort(key=seed_sort_key, reverse=True)
    return rows[: max(0, int(args.max_punctuation_seeds))]


def channel_specs(args: argparse.Namespace) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = [
        {
            "control_label": "coordinate_only",
            "control_family": "coordinate_baseline",
            "control_class": "coordinate_baseline",
            "candidate_group_kind": "coordinate_only",
            "candidate_group_factor": 1.0,
        }
    ]
    for group_kind in parse_csv(args.up_groups):
        for factor in parse_floats(args.up_factors):
            specs.append(
                {
                    "control_label": f"L{args.target_layer}_{group_kind}_up_{factor:g}",
                    "control_family": "punctuation_l39_channel_amplify",
                    "control_class": "candidate_gear",
                    "candidate_group_kind": group_kind,
                    "candidate_group_factor": float(factor),
                }
            )
    for group_kind in parse_csv(args.down_groups):
        for factor in parse_floats(args.down_factors):
            specs.append(
                {
                    "control_label": f"L{args.target_layer}_{group_kind}_down_{factor:g}",
                    "control_family": "punctuation_l39_channel_suppress",
                    "control_class": "candidate_gear",
                    "candidate_group_kind": group_kind,
                    "candidate_group_factor": float(factor),
                }
            )
    for group_kind in parse_csv(args.general_groups):
        for factor in parse_floats(args.general_factors):
            specs.append(
                {
                    "control_label": f"L{args.target_layer}_{group_kind}_general_{factor:g}",
                    "control_family": "punctuation_l39_channel_control",
                    "control_class": "candidate_gear",
                    "candidate_group_kind": group_kind,
                    "candidate_group_factor": float(factor),
                }
            )
    return specs


def surface_spec(base_spec: dict[str, Any], alpha: float, protocol_factor: float, span_kind: str) -> dict[str, Any]:
    spec = {
        "control_label": f"punct_{base_spec['control_label']}_route_{alpha:g}_protocol_{protocol_factor:g}",
        "control_family": base_spec["control_family"],
        "control_class": base_spec["control_class"],
        "route_alpha": float(alpha),
        "l4_factor_multiplier": 1.0,
        "protocol_span_kind": span_kind,
        "protocol_span_factor": float(protocol_factor),
        "candidate_group_kind": base_spec["candidate_group_kind"],
        "candidate_group_factor": float(base_spec["candidate_group_factor"]),
    }
    return spec


def logits_with_punctuation_candidate(
    model,
    device: torch.device,
    state: dict[str, Any],
    candidate_group: list[int],
    spec: dict[str, Any],
    target_layer: int,
) -> torch.Tensor | None:
    route_alpha = float(spec.get("route_alpha") or 1.0)
    route_delta = state["route_delta"] * route_alpha
    handles = p913.install_route_and_disentangle_hooks(
        model,
        route_delta,
        state["boundary_spec"],
        len(state["prompt_ids"]),
        len(state["prefix_ids"]),
        len(state["period_ids"]),
        state["l4_mlp_groups"],
    )
    if spec.get("candidate_group_kind") != "coordinate_only":
        handles.extend(
            p913.install_mlp_channel_group_scale(
                model,
                int(target_layer),
                candidate_group,
                float(spec.get("candidate_group_factor") or 1.0),
            )
        )
    span_start, span_end = p922.protocol_span_bounds(state, str(spec.get("protocol_span_kind")))
    handles.extend(
        p922.p909.install_attention_input_span_scale(
            model,
            0,
            int(span_start),
            int(span_end),
            float(spec.get("protocol_span_factor") or 1.0),
        )
    )
    if not handles:
        return None
    try:
        return p903.logits_plain(model, device, state["period_ids"])
    finally:
        for handle in handles:
            handle.remove()


def annotate_vs_coordinate_baseline(rows: list[dict[str, Any]]) -> None:
    baselines: dict[tuple[str, float, float], dict[str, Any]] = {}
    for row in rows:
        if row.get("candidate_group_kind") == "coordinate_only":
            baselines[(str(row.get("target_state_key")), float(row.get("route_alpha")), float(row.get("protocol_span_factor")))] = row
    for row in rows:
        base = baselines.get((str(row.get("target_state_key")), float(row.get("route_alpha")), float(row.get("protocol_span_factor"))))
        if base is None:
            continue
        row["coordinate_base_margin"] = base.get("patched_eos_margin_vs_blocker")
        row["coordinate_base_rank"] = base.get("patched_eos_rank")
        row["coordinate_base_top1"] = base.get("patched_eos_top1")
        row["coordinate_base_margin_nonnegative"] = base.get("patched_eos_margin_nonnegative")
        row["coordinate_base_strict_clean_candidate"] = base.get("strict_clean_candidate")
        row_margin = row.get("patched_eos_margin_vs_blocker")
        base_margin = base.get("patched_eos_margin_vs_blocker")
        row_rank = row.get("patched_eos_rank")
        base_rank = base.get("patched_eos_rank")
        row["margin_delta_vs_coordinate_base"] = None if row_margin is None or base_margin is None else float(row_margin - base_margin)
        row["rank_delta_vs_coordinate_base"] = None if row_rank is None or base_rank is None else int(row_rank) - int(base_rank)
        row["improved_margin_vs_coordinate_base"] = bool(
            row["margin_delta_vs_coordinate_base"] is not None and row["margin_delta_vs_coordinate_base"] > 0
        )
        row["worsened_margin_vs_coordinate_base"] = bool(
            row["margin_delta_vs_coordinate_base"] is not None and row["margin_delta_vs_coordinate_base"] < 0
        )
        row["new_margin_closure_vs_coordinate_base"] = bool(
            not base.get("patched_eos_margin_nonnegative") and row.get("patched_eos_margin_nonnegative")
        )
        row["new_top1_vs_coordinate_base"] = bool(not base.get("patched_eos_top1") and row.get("patched_eos_top1"))
        row["new_strict_vs_coordinate_base"] = bool(
            not base.get("strict_clean_candidate") and row.get("strict_clean_candidate")
        )


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    candidate_rows = [row for row in rows if row.get("candidate_group_kind") != "coordinate_only"]
    baseline_rows = [row for row in rows if row.get("candidate_group_kind") == "coordinate_only"]
    return {
        "rows": len(rows),
        "coordinate_baseline_rows": len(baseline_rows),
        "candidate_rows": len(candidate_rows),
        "unique_states": len({row.get("target_state_key") for row in rows}),
        "unique_cases": len({row.get("target_case_id") for row in rows}),
        "top1": sum(1 for row in rows if row.get("patched_eos_top1")),
        "margin_nonnegative": sum(1 for row in rows if row.get("patched_eos_margin_nonnegative")),
        "strict_clean_candidate": sum(1 for row in rows if row.get("strict_clean_candidate")),
        "candidate_top1": sum(1 for row in candidate_rows if row.get("patched_eos_top1")),
        "candidate_margin_nonnegative": sum(1 for row in candidate_rows if row.get("patched_eos_margin_nonnegative")),
        "candidate_strict_clean_candidate": sum(1 for row in candidate_rows if row.get("strict_clean_candidate")),
        "improved_margin_vs_coordinate_base": sum(1 for row in candidate_rows if row.get("improved_margin_vs_coordinate_base")),
        "worsened_margin_vs_coordinate_base": sum(1 for row in candidate_rows if row.get("worsened_margin_vs_coordinate_base")),
        "new_margin_closure_vs_coordinate_base": sum(1 for row in candidate_rows if row.get("new_margin_closure_vs_coordinate_base")),
        "new_top1_vs_coordinate_base": sum(1 for row in candidate_rows if row.get("new_top1_vs_coordinate_base")),
        "new_strict_vs_coordinate_base": sum(1 for row in candidate_rows if row.get("new_strict_vs_coordinate_base")),
        "median_margin_delta_vs_coordinate_base": median([row.get("margin_delta_vs_coordinate_base") for row in candidate_rows]),
        "mean_margin_delta_vs_coordinate_base": mean([row.get("margin_delta_vs_coordinate_base") for row in candidate_rows]),
        "median_patched_margin": median([row.get("patched_eos_margin_vs_blocker") for row in rows]),
        "target_state_coverage_top1": len({row.get("target_state_key") for row in rows if row.get("patched_eos_top1")}),
        "target_state_coverage_margin": len(
            {row.get("target_state_key") for row in rows if row.get("patched_eos_margin_nonnegative")}
        ),
        "target_state_coverage_strict": len({row.get("target_state_key") for row in rows if row.get("strict_clean_candidate")}),
        "patched_blocker_class_distribution": dict(Counter(p926.blocker_class(row.get("patched_blocker_token")) for row in rows)),
    }


def summarize_by(rows: list[dict[str, Any]], keys: list[str], limit: int = 160) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[tuple(str(row.get(key)) for key in keys)].append(row)
    out = []
    for key_tuple, vals in buckets.items():
        summary = summarize_rows(vals)
        for key, value in zip(keys, key_tuple):
            summary[key] = value
        out.append(summary)
    out.sort(
        key=lambda row: (
            row.get("new_strict_vs_coordinate_base") or 0,
            row.get("new_top1_vs_coordinate_base") or 0,
            row.get("new_margin_closure_vs_coordinate_base") or 0,
            row.get("candidate_top1") or 0,
            row.get("improved_margin_vs_coordinate_base") or 0,
            row.get("mean_margin_delta_vs_coordinate_base") or -9999,
        ),
        reverse=True,
    )
    return out[:limit]


def evidence_label(selected_count: int, rows: list[dict[str, Any]]) -> str:
    overall = summarize_rows(rows)
    if selected_count == 0:
        return "no_punctuation_period_seeds"
    if overall["new_strict_vs_coordinate_base"] > 0:
        return "punctuation_specific_strict_candidate_found"
    if overall["new_top1_vs_coordinate_base"] > 0 or overall["new_margin_closure_vs_coordinate_base"] > 0:
        return "punctuation_specific_closure_candidate_found"
    if overall["candidate_top1"] > 0 or overall["candidate_margin_nonnegative"] > 0:
        return "punctuation_coordinate_closure_not_channel_specific"
    if overall["improved_margin_vs_coordinate_base"] > 0:
        return "punctuation_specific_margin_movement_only"
    return "no_punctuation_specific_candidate_found"


def eval_model(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = RESULT_ROOT / args.round_name
    out_dir.mkdir(parents=True, exist_ok=True)
    selected = select_punctuation_seeds(args.model, args)
    coords = parse_coordinate_pairs(args.coordinate_pairs)
    specs = channel_specs(args)
    if args.dry_run or not selected:
        payload = {
            "phase": PHASE,
            "title": "Punctuation-Specific Protocol Gear Search",
            "model": args.model,
            "status": "dry_run" if args.dry_run else "no_punctuation_period_seeds",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "selected_punctuation_seeds": len(selected),
            "coordinate_count": len(coords),
            "channel_spec_count": len(specs),
            "expected_rows_if_all_reconstructed": len(selected) * len(coords) * len(specs),
            "overall": summarize_rows([]),
            "by_group": [],
            "evidence_label": "no_punctuation_period_seeds" if not selected else "dry_run",
        }
        p846.write_json(out_dir / f"phase928_{args.model}_summary.json", payload)
        p846.write_jsonl(out_dir / f"phase928_{args.model}_rows.jsonl", [])
        p846.write_jsonl(out_dir / f"phase928_{args.model}_selected_seeds.jsonl", selected)
        print(json.dumps({"phase": PHASE, "model": args.model, "status": payload["status"], "selected": len(selected)}, ensure_ascii=False, indent=2), flush=True)
        return payload

    case_map = {str(case.get("case_id")): case for case in p885.extended_cases()}
    model = None
    tokenizer = None
    states: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
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
            log(f"{args.model}/{args.round_name}: reconstructed_punctuation_seed={idx}/{len(selected)} kept={len(states)}")
        for state_idx, state in enumerate(states, 1):
            for alpha, protocol_factor in coords:
                for base_spec in specs:
                    spec = surface_spec(base_spec, alpha, protocol_factor, args.protocol_span_kind)
                    group_kind = str(spec.get("candidate_group_kind"))
                    candidate_group = [] if group_kind == "coordinate_only" else state["channel_groups"].get(group_kind, [])
                    if group_kind != "coordinate_only" and not candidate_group:
                        continue
                    patched_logits = logits_with_punctuation_candidate(
                        model,
                        device,
                        state,
                        candidate_group,
                        spec,
                        int(args.target_layer),
                    )
                    if patched_logits is None:
                        continue
                    row = p922.row_from_logits(
                        tokenizer,
                        state,
                        candidate_group,
                        float(spec.get("candidate_group_factor") or 1.0),
                        spec,
                        patched_logits,
                        groups,
                    )
                    row["phase"] = PHASE
                    row["row_kind"] = "phase928_punctuation_specific_protocol_gear_row"
                    row["phase925_surface_state_key"] = state["source_row"].get("surface_state_key")
                    row["phase925_seed_blocker_token"] = state["source_row"].get("patched_blocker_token")
                    row["phase925_seed_blocker_class"] = p926.blocker_class(state["source_row"].get("patched_blocker_token"))
                    row["phase925_group_kind"] = state["source_row"].get("group_kind")
                    row["phase925_factor"] = state["source_row"].get("factor")
                    row["candidate_group_kind"] = group_kind
                    row["candidate_group_factor"] = float(spec.get("candidate_group_factor") or 1.0)
                    row["candidate_group_size"] = len(candidate_group)
                    rows.append(row)
            if state_idx % max(1, int(args.log_every)) == 0 or state_idx == len(states):
                log(f"{args.model}/{args.round_name}: punctuation_gear_state={state_idx}/{len(states)} rows={len(rows)}")
    finally:
        if model is not None:
            p862.p844.p828.release_model(model)
        if tokenizer is not None:
            del tokenizer
        del states
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    annotate_vs_coordinate_baseline(rows)
    label = evidence_label(len(selected), rows)
    payload = {
        "phase": PHASE,
        "title": "Punctuation-Specific Protocol Gear Search",
        "model": args.model,
        "status": "complete",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "attn_implementation": attn_impl,
        "selected_punctuation_seeds": len(selected),
        "coordinate_count": len(coords),
        "channel_spec_count": len(specs),
        "expected_rows_if_all_reconstructed": len(selected) * len(coords) * len(specs),
        "overall": summarize_rows(rows),
        "by_group": summarize_by(rows, ["candidate_group_kind"], limit=120),
        "by_group_factor": summarize_by(rows, ["candidate_group_kind", "candidate_group_factor"], limit=200),
        "by_coordinate": summarize_by(rows, ["route_alpha", "protocol_span_factor"], limit=120),
        "by_group_coordinate": summarize_by(rows, ["candidate_group_kind", "route_alpha", "protocol_span_factor"], limit=240),
        "top_candidate_rows": [
            row
            for row in sorted(
                [r for r in rows if r.get("candidate_group_kind") != "coordinate_only"],
                key=lambda r: (
                    r.get("new_strict_vs_coordinate_base") or 0,
                    r.get("new_top1_vs_coordinate_base") or 0,
                    r.get("new_margin_closure_vs_coordinate_base") or 0,
                    r.get("patched_eos_top1") or 0,
                    r.get("improved_margin_vs_coordinate_base") or 0,
                    r.get("margin_delta_vs_coordinate_base") or -9999,
                ),
                reverse=True,
            )[:160]
        ],
        "new_closure_rows": [
            row
            for row in rows
            if row.get("new_top1_vs_coordinate_base") or row.get("new_margin_closure_vs_coordinate_base")
        ],
        "evidence_label": label,
        "boundary": (
            "Phase928 runs new forward interventions on punctuation_period seeds. It compares punctuation-specific "
            "L39 channel groups against a same-coordinate route/protocol baseline, so route/protocol movement is not "
            "misread as a new gear effect."
        ),
    }
    p846.write_json(out_dir / f"phase928_{args.model}_summary.json", payload)
    p846.write_jsonl(out_dir / f"phase928_{args.model}_rows.jsonl", rows)
    p846.write_jsonl(out_dir / f"phase928_{args.model}_selected_seeds.jsonl", selected)
    print(
        json.dumps(
            {
                "phase": PHASE,
                "model": args.model,
                "selected": len(selected),
                "rows": len(rows),
                "overall": payload["overall"],
                "evidence_label": label,
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
    top_groups = []
    top_rows = []
    for model_name in MODELS:
        summary = read_json(out_dir / f"phase928_{model_name}_summary.json")
        if not summary:
            continue
        summaries.append(summary)
        evidence[str(summary.get("evidence_label"))] += 1
        scalar["selected_punctuation_seeds"] += int(summary.get("selected_punctuation_seeds") or 0)
        scalar["expected_rows_if_all_reconstructed"] += int(summary.get("expected_rows_if_all_reconstructed") or 0)
        overall = summary.get("overall") or {}
        for key, value in overall.items():
            if isinstance(value, int):
                scalar[f"overall_{key}"] += int(value)
        for row in summary.get("by_group_factor") or []:
            item = dict(row)
            item["model"] = model_name
            top_groups.append(item)
        for row in summary.get("top_candidate_rows") or []:
            item = dict(row)
            item["model"] = model_name
            top_rows.append(item)
    top_groups.sort(
        key=lambda row: (
            row.get("new_strict_vs_coordinate_base") or 0,
            row.get("new_top1_vs_coordinate_base") or 0,
            row.get("new_margin_closure_vs_coordinate_base") or 0,
            row.get("candidate_top1") or 0,
            row.get("improved_margin_vs_coordinate_base") or 0,
            row.get("mean_margin_delta_vs_coordinate_base") or -9999,
        ),
        reverse=True,
    )
    top_rows.sort(
        key=lambda row: (
            row.get("new_strict_vs_coordinate_base") or 0,
            row.get("new_top1_vs_coordinate_base") or 0,
            row.get("new_margin_closure_vs_coordinate_base") or 0,
            row.get("patched_eos_top1") or 0,
            row.get("improved_margin_vs_coordinate_base") or 0,
            row.get("margin_delta_vs_coordinate_base") or -9999,
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
        "top_group_factors": top_groups[:160],
        "top_candidate_rows": top_rows[:160],
    }
    p846.write_json(out_dir / "phase928_cross_model_summary.json", payload)
    write_markdown(out_dir / "phase928_cross_model_summary.md", payload)
    return payload


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Phase 928 punctuation-specific protocol gear search",
        "",
        "## Overall",
        "",
        f"- models: {', '.join(payload.get('models') or [])}",
    ]
    for key, value in (payload.get("overall_scalar") or {}).items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Evidence", ""])
    for key, value in (payload.get("evidence_label_counts") or {}).items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Top Group Factors", ""])
    lines.append(
        "| model | group | factor | rows | top1 | margin | strict | improved | new margin | new top1 | mean delta |"
    )
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in payload.get("top_group_factors") or []:
        row = {
            "model": "",
            "candidate_group_kind": "",
            "candidate_group_factor": "",
            "rows": 0,
            "candidate_top1": 0,
            "candidate_margin_nonnegative": 0,
            "candidate_strict_clean_candidate": 0,
            "improved_margin_vs_coordinate_base": 0,
            "new_margin_closure_vs_coordinate_base": 0,
            "new_top1_vs_coordinate_base": 0,
            "mean_margin_delta_vs_coordinate_base": None,
            **row,
        }
        lines.append(
            "| {model} | {candidate_group_kind} | {candidate_group_factor} | {rows} | {candidate_top1} | {candidate_margin_nonnegative} | {candidate_strict_clean_candidate} | {improved_margin_vs_coordinate_base} | {new_margin_closure_vs_coordinate_base} | {new_top1_vs_coordinate_base} | {mean_margin_delta_vs_coordinate_base} |".format(
                **row
            )
        )
    lines.extend(["", "## Top Candidate Rows", ""])
    lines.append(
        "| model | state | case | group | factor | alpha | protocol | rank | margin | base margin | delta | top1 | strict |"
    )
    lines.append("| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |")
    for row in payload.get("top_candidate_rows") or []:
        row = {
            "model": "",
            "target_state_key": "",
            "target_case_id": "",
            "candidate_group_kind": "",
            "candidate_group_factor": "",
            "route_alpha": "",
            "protocol_span_factor": "",
            "patched_eos_rank": "",
            "patched_eos_margin_vs_blocker": "",
            "coordinate_base_margin": "",
            "margin_delta_vs_coordinate_base": "",
            "patched_eos_top1": "",
            "strict_clean_candidate": "",
            **row,
        }
        lines.append(
            "| {model} | {target_state_key} | {target_case_id} | {candidate_group_kind} | {candidate_group_factor} | {route_alpha} | {protocol_span_factor} | {patched_eos_rank} | {patched_eos_margin_vs_blocker} | {coordinate_base_margin} | {margin_delta_vs_coordinate_base} | {patched_eos_top1} | {strict_clean_candidate} |".format(
                **row
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--round-name", default="punctuation_specific_protocol_gear_search")
    parser.add_argument("--phase926-round", default="generalized_route_protocol_surface_validation")
    parser.add_argument("--max-punctuation-seeds", type=int, default=12)
    parser.add_argument("--coordinate-pairs", default="1.0:1.0,0.875:1.1,1.25:1.1,0.875:0.85,1.25:0.85,1.375:0.85,1.375:0.9")
    parser.add_argument("--protocol-span-kind", default="last8_before_period")
    parser.add_argument("--target-layer", type=int, default=39)
    parser.add_argument("--max-prefix-tokens", type=int, default=5)
    parser.add_argument("--scale-up-factor", type=float, default=2.0)
    parser.add_argument("--l4-candidate-pool", type=int, default=512)
    parser.add_argument("--channel-candidate-pool", type=int, default=768)
    parser.add_argument("--band-size", type=int, default=32)
    parser.add_argument("--up-groups", default="eos_support_64,margin_support_pos_64")
    parser.add_argument("--up-factors", default="1.25,1.5,2.0")
    parser.add_argument("--down-groups", default="a_blocker_support_64,a_logit_support_64,margin_support_neg_64,band_blocker_support_64")
    parser.add_argument("--down-factors", default="0.0,0.25,0.5,0.75")
    parser.add_argument("--general-groups", default="top_abs_64,low_abs_64")
    parser.add_argument("--general-factors", default="0.5,1.5")
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
