#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import sys
from collections import Counter
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
import phase926_generalized_route_protocol_surface_validation as p926  # noqa: E402
import phase928_punctuation_specific_protocol_gear_search as p928  # noqa: E402
import phase929_punctuation_margin_gear_holdout_validation as p929  # noqa: E402


PHASE = 931
MODELS = ["qwen3", "glm4", "deepseek7b"]
RESULT_ROOT = Path("tests/result/phase931_fixed_punctuation_margin_gear_causal_validation")
PHASE930_ROOT = Path("tests/result/phase930_natural_gate_strict_clean_transition_audit")


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def consensus_groups(model_name: str, args: argparse.Namespace) -> dict[str, list[int]]:
    path = PHASE930_ROOT / args.phase930_round / f"phase930_{model_name}_state_features.jsonl"
    rows = read_jsonl(path)
    groups = [[int(x) for x in row.get("margin_support_pos_64_channels") or []] for row in rows]
    groups = [group for group in groups if group]
    if not groups:
        return {}
    counter: Counter[int] = Counter()
    for group in groups:
        counter.update(set(group))
    intersection = set(groups[0])
    for group in groups:
        intersection &= set(group)
    top64 = [int(ch) for ch, _count in counter.most_common(64)]
    top31 = [int(ch) for ch, _count in counter.most_common(max(1, len(intersection)))]
    half = [int(ch) for ch, count in counter.most_common() if count >= len(groups) / 2]
    return {
        "fixed_intersection_all": sorted(int(x) for x in intersection),
        "fixed_topfreq_31": top31,
        "fixed_topfreq_64": top64,
        "fixed_half_or_more": half[:64],
    }


def build_specs(args: argparse.Namespace, fixed_groups: dict[str, list[int]]) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = [
        {
            "control_label": "coordinate_only",
            "control_family": "coordinate_baseline",
            "control_class": "coordinate_baseline",
            "candidate_group_kind": "coordinate_only",
            "candidate_group_factor": 1.0,
        }
    ]
    for factor in p928.parse_floats(args.factors):
        specs.append(
            {
                "control_label": f"state_specific_margin_support_pos_64_{factor:g}",
                "control_family": "state_specific_margin_support",
                "control_class": "state_specific_reference",
                "candidate_group_kind": "state_specific_margin_support_pos_64",
                "candidate_group_factor": float(factor),
            }
        )
        for name, group in fixed_groups.items():
            if not group:
                continue
            specs.append(
                {
                    "control_label": f"{name}_{factor:g}",
                    "control_family": "fixed_punctuation_margin_support",
                    "control_class": "fixed_consensus_candidate",
                    "candidate_group_kind": name,
                    "candidate_group_factor": float(factor),
                }
            )
    return specs


def group_for_spec(state: dict[str, Any], fixed_groups: dict[str, list[int]], group_kind: str) -> list[int]:
    if group_kind == "coordinate_only":
        return []
    if group_kind == "state_specific_margin_support_pos_64":
        return [int(x) for x in (state.get("channel_groups") or {}).get("margin_support_pos_64", [])]
    return [int(x) for x in fixed_groups.get(group_kind, [])]


def evidence_label(selected_count: int, rows: list[dict[str, Any]]) -> str:
    if selected_count <= 0:
        return "no_punctuation_period_seeds"
    fixed_rows = [row for row in rows if str(row.get("control_class")) == "fixed_consensus_candidate"]
    if any(row.get("new_strict_vs_coordinate_base") for row in fixed_rows):
        return "fixed_punctuation_margin_gear_strict_positive"
    if any(row.get("new_top1_vs_coordinate_base") or row.get("new_margin_closure_vs_coordinate_base") for row in fixed_rows):
        return "fixed_punctuation_margin_gear_causal_positive"
    reference_rows = [row for row in rows if str(row.get("control_class")) == "state_specific_reference"]
    if any(row.get("new_top1_vs_coordinate_base") or row.get("new_margin_closure_vs_coordinate_base") for row in reference_rows):
        return "state_specific_positive_fixed_negative"
    return "no_margin_gear_positive"


def eval_model(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = RESULT_ROOT / args.round_name
    out_dir.mkdir(parents=True, exist_ok=True)
    selected = p929.select_punctuation_seeds(args.model, args)
    coords = p928.parse_coordinate_pairs(args.coordinate_pairs)
    fixed_groups = consensus_groups(args.model, args)
    specs = build_specs(args, fixed_groups)
    if args.dry_run or not selected or not fixed_groups:
        status = "dry_run" if args.dry_run else ("no_punctuation_period_seeds" if not selected else "no_phase930_fixed_groups")
        payload = {
            "phase": PHASE,
            "title": "Fixed Punctuation Margin Gear Causal Validation",
            "model": args.model,
            "status": status,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "selected_punctuation_seeds": len(selected),
            "coordinate_count": len(coords),
            "channel_spec_count": len(specs),
            "expected_rows_if_all_reconstructed": len(selected) * len(coords) * len(specs),
            "fixed_group_sizes": {name: len(group) for name, group in fixed_groups.items()},
            "overall": p928.summarize_rows([]),
            "evidence_label": status,
        }
        p846.write_json(out_dir / f"phase931_{args.model}_summary.json", payload)
        p846.write_jsonl(out_dir / f"phase931_{args.model}_rows.jsonl", [])
        p846.write_jsonl(out_dir / f"phase931_{args.model}_selected_seeds.jsonl", selected)
        print(json.dumps({"phase": PHASE, "model": args.model, "status": status, "selected": len(selected)}, ensure_ascii=False, indent=2), flush=True)
        return payload

    case_map = {str(case.get("case_id")): case for case in p885.extended_cases()}
    rows: list[dict[str, Any]] = []
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
            log(f"{args.model}/{args.round_name}: reconstructed_fixed_seed={idx}/{len(selected)} kept={len(states)}")
        for state_idx, state in enumerate(states, 1):
            state_key = str(state["source_row"].get("surface_state_key"))
            for alpha, protocol_factor in coords:
                for base_spec in specs:
                    spec = p928.surface_spec(base_spec, alpha, protocol_factor, args.protocol_span_kind)
                    group_kind = str(spec.get("candidate_group_kind"))
                    candidate_group = group_for_spec(state, fixed_groups, group_kind)
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
                    row["row_kind"] = "phase931_fixed_punctuation_margin_gear_row"
                    row["phase925_surface_state_key"] = state_key
                    row["candidate_group_kind"] = group_kind
                    row["candidate_group_factor"] = float(spec.get("candidate_group_factor") or 1.0)
                    row["candidate_group_size"] = len(candidate_group)
                    row["phase925_group_kind"] = state["source_row"].get("group_kind")
                    row["phase925_factor"] = state["source_row"].get("factor")
                    rows.append(row)
            if state_idx % max(1, int(args.log_every)) == 0 or state_idx == len(states):
                log(f"{args.model}/{args.round_name}: fixed_state={state_idx}/{len(states)} rows={len(rows)}")
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
    label = evidence_label(len(selected), rows)
    payload = {
        "phase": PHASE,
        "title": "Fixed Punctuation Margin Gear Causal Validation",
        "model": args.model,
        "status": "complete",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "attn_implementation": attn_impl,
        "selected_punctuation_seeds": len(selected),
        "coordinate_count": len(coords),
        "channel_spec_count": len(specs),
        "expected_rows_if_all_reconstructed": len(selected) * len(coords) * len(specs),
        "fixed_group_sizes": {name: len(group) for name, group in fixed_groups.items()},
        "overall": p928.summarize_rows(rows),
        "by_group_factor": p928.summarize_by(rows, ["candidate_group_kind", "candidate_group_factor"], limit=200),
        "by_group_case": p928.summarize_by(rows, ["candidate_group_kind", "target_case_id"], limit=240),
        "evidence_label": label,
        "boundary": "fixed channel causal validation only; not natural gate closure",
    }
    p846.write_json(out_dir / f"phase931_{args.model}_summary.json", payload)
    p846.write_jsonl(out_dir / f"phase931_{args.model}_rows.jsonl", rows)
    p846.write_jsonl(out_dir / f"phase931_{args.model}_selected_seeds.jsonl", selected)
    print(json.dumps({"phase": PHASE, "model": args.model, "status": label, "overall": payload["overall"]}, ensure_ascii=False, indent=2), flush=True)
    return payload


def summarize_round(round_name: str) -> dict[str, Any]:
    out_dir = RESULT_ROOT / round_name
    summaries = [read_json(out_dir / f"phase931_{model}_summary.json") for model in MODELS]
    summaries = [summary for summary in summaries if summary]
    evidence_counts: Counter[str] = Counter()
    overall_scalar: dict[str, Any] = {}
    group_rows: list[dict[str, Any]] = []
    for summary in summaries:
        evidence_counts[str(summary.get("evidence_label"))] += 1
        overall_scalar["selected_punctuation_seeds"] = overall_scalar.get("selected_punctuation_seeds", 0) + int(summary.get("selected_punctuation_seeds") or 0)
        overall_scalar["expected_rows_if_all_reconstructed"] = overall_scalar.get("expected_rows_if_all_reconstructed", 0) + int(summary.get("expected_rows_if_all_reconstructed") or 0)
        for key, value in (summary.get("overall") or {}).items():
            if isinstance(value, int):
                overall_scalar[f"overall_{key}"] = overall_scalar.get(f"overall_{key}", 0) + value
        for row in summary.get("by_group_factor") or []:
            item = dict(row)
            item["model"] = summary.get("model")
            group_rows.append(item)
    group_rows.sort(key=lambda row: (int(row.get("new_top1_vs_coordinate_base") or 0), int(row.get("top1") or 0), float(row.get("mean_margin_delta_vs_coordinate_base") or -999)), reverse=True)
    payload = {
        "phase": PHASE,
        "round": round_name,
        "status": "complete" if summaries else "missing",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "models": [summary.get("model") for summary in summaries],
        "overall_scalar": overall_scalar,
        "evidence_label_counts": dict(evidence_counts),
        "top_group_factor_rows": group_rows[:120],
        "model_summaries": summaries,
    }
    p846.write_json(out_dir / "phase931_cross_model_summary.json", payload)
    write_summary_md(out_dir / "phase931_cross_model_summary.md", payload)
    return payload


def write_summary_md(path: Path, payload: dict[str, Any]) -> None:
    lines = ["# Phase 931 fixed punctuation margin gear causal validation", "", "## Overall", ""]
    for key, value in sorted((payload.get("overall_scalar") or {}).items()):
        lines.append(f"- {key}: {value}")
    lines += ["", "## Evidence", ""]
    for key, value in sorted((payload.get("evidence_label_counts") or {}).items()):
        lines.append(f"- {key}: {value}")
    lines += ["", "## Top Group Factor Rows", ""]
    lines.append("| model | group | factor | rows | top1 | margin | strict | new top1 | new margin | states | mean delta |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in payload.get("top_group_factor_rows") or []:
        lines.append(
            "| {model} | {candidate_group_kind} | {candidate_group_factor} | {rows} | {top1} | {margin_nonnegative} | {strict_clean_candidate} | {new_top1_vs_coordinate_base} | {new_margin_closure_vs_coordinate_base} | {target_state_coverage_top1} | {mean_margin_delta_vs_coordinate_base} |".format(
                **row
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--round-name", default="fixed_punctuation_margin_gear_causal_validation")
    parser.add_argument("--phase925-round", default="response_surface_generalization_dataset_expansion")
    parser.add_argument("--phase930-round", default="natural_gate_strict_clean_transition_audit")
    parser.add_argument("--seed-source", choices=["selected", "candidate"], default="selected")
    parser.add_argument("--max-punctuation-seeds", type=int, default=30)
    parser.add_argument("--max-per-case", type=int, default=10)
    parser.add_argument("--coordinate-pairs", default="1.0:1.0,0.875:1.1,1.25:1.1,0.875:0.85,1.25:0.85,1.375:0.85,1.375:0.9")
    parser.add_argument("--factors", default="2.1,2.25")
    parser.add_argument("--protocol-span-kind", default="last8_before_period")
    parser.add_argument("--target-layer", type=int, default=39)
    parser.add_argument("--max-prefix-tokens", type=int, default=5)
    parser.add_argument("--scale-up-factor", type=float, default=2.0)
    parser.add_argument("--l4-candidate-pool", type=int, default=512)
    parser.add_argument("--channel-candidate-pool", type=int, default=768)
    parser.add_argument("--band-size", type=int, default=32)
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
