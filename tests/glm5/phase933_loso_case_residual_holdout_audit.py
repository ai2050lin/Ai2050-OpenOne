#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
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
import phase926_generalized_route_protocol_surface_validation as p926  # noqa: E402
import phase928_punctuation_specific_protocol_gear_search as p928  # noqa: E402
import phase929_punctuation_margin_gear_holdout_validation as p929  # noqa: E402
import phase931_fixed_punctuation_margin_gear_causal_validation as p931  # noqa: E402
import phase932_fixed_gear_repair_case_residual_audit as p932  # noqa: E402


PHASE = 933
MODELS = ["qwen3", "glm4", "deepseek7b"]
RESULT_ROOT = Path("tests/result/phase933_loso_case_residual_holdout_audit")
PHASE930_ROOT = Path("tests/result/phase930_natural_gate_strict_clean_transition_audit")


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


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
    group_kinds = [
        "fixed_topfreq_64",
        "loso_case_inter_residual_only",
        "loso_case_union_residual_only",
        "fixed_plus_loso_case_inter_residual",
        "fixed_plus_loso_case_union_residual",
        "state_specific_margin_support_pos_64",
    ]
    for factor in p928.parse_floats(args.factors):
        for group_kind in group_kinds:
            control_class = "state_specific_reference" if group_kind == "state_specific_margin_support_pos_64" else "loso_repair_candidate"
            specs.append(
                {
                    "control_label": f"{group_kind}_{factor:g}",
                    "control_family": "loso_case_residual_holdout",
                    "control_class": control_class,
                    "candidate_group_kind": group_kind,
                    "candidate_group_factor": float(factor),
                }
            )
    return specs


def loso_inventory(model_name: str, args: argparse.Namespace, fixed_top64: list[int]) -> dict[str, Any]:
    path = PHASE930_ROOT / args.phase930_round / f"phase930_{model_name}_state_features.jsonl"
    rows = read_jsonl(path)
    rows_by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        group = p932.channels_from_feature_row(row)
        if group:
            rows_by_case[str(row.get("target_case_id"))].append(row)
    fixed = set(int(x) for x in fixed_top64)
    by_state: dict[str, dict[str, Any]] = {}
    by_case_summary: dict[str, Any] = {}
    for case_id, case_rows in sorted(rows_by_case.items()):
        residual_sizes = []
        for target in case_rows:
            target_key = str(target.get("target_state_key"))
            holdout_rows = [row for row in case_rows if str(row.get("target_state_key")) != target_key]
            groups = [p932.channels_from_feature_row(row) for row in holdout_rows]
            if not groups:
                by_state[target_key] = {
                    "target_case_id": case_id,
                    "holdout_state_count": 0,
                    "intersection_minus_fixed_top64": [],
                    "union_minus_fixed_top64": [],
                }
                continue
            intersection = set(groups[0])
            union: set[int] = set()
            for group in groups:
                unique = set(group)
                intersection &= unique
                union |= unique
            inter_res = sorted(int(ch) for ch in intersection - fixed)
            union_res = sorted(int(ch) for ch in union - fixed)
            residual_sizes.append((len(inter_res), len(union_res)))
            by_state[target_key] = {
                "target_case_id": case_id,
                "holdout_state_count": len(holdout_rows),
                "intersection_size": len(intersection),
                "union_size": len(union),
                "intersection_minus_fixed_top64": inter_res,
                "union_minus_fixed_top64": union_res,
            }
        by_case_summary[case_id] = {
            "state_count": len(case_rows),
            "holdout_state_count_each": max(0, len(case_rows) - 1),
            "inter_residual_size_min": min((x[0] for x in residual_sizes), default=0),
            "inter_residual_size_max": max((x[0] for x in residual_sizes), default=0),
            "union_residual_size_min": min((x[1] for x in residual_sizes), default=0),
            "union_residual_size_max": max((x[1] for x in residual_sizes), default=0),
        }
    return {
        "phase930_feature_rows": len(rows),
        "fixed_topfreq64_size": len(fixed_top64),
        "by_case_summary": by_case_summary,
        "by_state": by_state,
    }


def group_for_spec(
    state: dict[str, Any],
    fixed_top64: list[int],
    inventory: dict[str, Any],
    group_kind: str,
) -> tuple[list[int], str]:
    state_key = str(state["source_row"].get("surface_state_key"))
    item = (inventory.get("by_state") or {}).get(state_key) or {}
    inter = [int(x) for x in item.get("intersection_minus_fixed_top64") or []]
    union = [int(x) for x in item.get("union_minus_fixed_top64") or []]
    if group_kind == "coordinate_only":
        return [], "coordinate_only"
    if group_kind == "fixed_topfreq_64":
        return fixed_top64, "fixed_topfreq_64"
    if group_kind == "loso_case_inter_residual_only":
        return inter, f"{state_key}:loso_intersection_minus_fixed_top64"
    if group_kind == "loso_case_union_residual_only":
        return union, f"{state_key}:loso_union_minus_fixed_top64"
    if group_kind == "fixed_plus_loso_case_inter_residual":
        return p932.ordered_union(fixed_top64, inter), f"fixed_topfreq_64+{state_key}:loso_intersection_minus_fixed_top64"
    if group_kind == "fixed_plus_loso_case_union_residual":
        return p932.ordered_union(fixed_top64, union), f"fixed_topfreq_64+{state_key}:loso_union_minus_fixed_top64"
    if group_kind == "state_specific_margin_support_pos_64":
        return [int(x) for x in (state.get("channel_groups") or {}).get("margin_support_pos_64", [])], "state_specific_margin_support_pos_64"
    return [], "unknown"


def coverage(rows: list[dict[str, Any]], group_kind: str, factor: float | None = None) -> int:
    return p932.state_coverage(rows, group_kind, factor)


def repair_comparison(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    out = []
    cases = sorted({str(row.get("target_case_id")) for row in rows if row.get("target_case_id")})
    group_kinds = [
        "fixed_topfreq_64",
        "loso_case_inter_residual_only",
        "loso_case_union_residual_only",
        "fixed_plus_loso_case_inter_residual",
        "fixed_plus_loso_case_union_residual",
        "state_specific_margin_support_pos_64",
    ]
    for factor in p928.parse_floats(args.factors):
        for group_kind in group_kinds:
            item = {
                "candidate_group_kind": group_kind,
                "candidate_group_factor": float(factor),
                "state_coverage_all": coverage(rows, group_kind, factor),
            }
            for case_id in cases:
                item[f"state_coverage_{case_id}"] = p932.state_coverage(rows, group_kind, factor, case_id)
            out.append(item)
    out.sort(key=lambda row: (row["state_coverage_all"], row["candidate_group_factor"]), reverse=True)
    return out


def evidence_label(selected_count: int, rows: list[dict[str, Any]]) -> str:
    if selected_count <= 0:
        return "no_punctuation_period_seeds"
    repair_rows = [row for row in rows if str(row.get("control_class")) == "loso_repair_candidate"]
    if any(row.get("new_strict_vs_coordinate_base") for row in repair_rows):
        return "loso_case_residual_strict_clean_positive"
    fixed_cov = coverage(rows, "fixed_topfreq_64", 2.25)
    loso_cov = max(
        coverage(rows, "fixed_plus_loso_case_inter_residual", 2.25),
        coverage(rows, "fixed_plus_loso_case_union_residual", 2.25),
    )
    reference_cov = coverage(rows, "state_specific_margin_support_pos_64", 2.25)
    if loso_cov >= reference_cov and loso_cov > fixed_cov:
        return "loso_case_residual_matches_state_specific_without_strict_clean"
    if loso_cov > fixed_cov:
        return "loso_case_residual_partial_positive_without_strict_clean"
    if reference_cov > fixed_cov:
        return "loso_case_residual_negative_state_specific_remains_needed"
    return "no_loso_case_residual_gain"


def eval_model(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = RESULT_ROOT / args.round_name
    out_dir.mkdir(parents=True, exist_ok=True)
    selected = p929.select_punctuation_seeds(args.model, args)
    coords = p928.parse_coordinate_pairs(args.coordinate_pairs)
    fixed_groups = p931.consensus_groups(args.model, args)
    fixed_top64 = [int(x) for x in fixed_groups.get("fixed_topfreq_64") or []]
    inventory = loso_inventory(args.model, args, fixed_top64)
    specs = build_specs(args)
    if args.dry_run or not selected or not fixed_top64:
        status = "dry_run" if args.dry_run else ("no_punctuation_period_seeds" if not selected else "no_phase930_fixed_topfreq64")
        payload = {
            "phase": PHASE,
            "title": "LOSO Case Residual Holdout Audit",
            "model": args.model,
            "status": status,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "selected_punctuation_seeds": len(selected),
            "coordinate_count": len(coords),
            "channel_spec_count": len(specs),
            "expected_rows_if_all_reconstructed": len(selected) * len(coords) * len(specs),
            "fixed_group_sizes": {name: len(group) for name, group in fixed_groups.items()},
            "loso_residual_inventory": {k: v for k, v in inventory.items() if k != "by_state"},
            "overall": p928.summarize_rows([]),
            "repair_comparison": [],
            "evidence_label": status,
        }
        p846.write_json(out_dir / f"phase933_{args.model}_summary.json", payload)
        p846.write_jsonl(out_dir / f"phase933_{args.model}_rows.jsonl", [])
        p846.write_jsonl(out_dir / f"phase933_{args.model}_selected_seeds.jsonl", selected)
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
            log(f"{args.model}/{args.round_name}: reconstructed_loso_seed={idx}/{len(selected)} kept={len(states)}")
        for state_idx, state in enumerate(states, 1):
            state_key = str(state["source_row"].get("surface_state_key"))
            for alpha, protocol_factor in coords:
                for base_spec in specs:
                    spec = p928.surface_spec(base_spec, alpha, protocol_factor, args.protocol_span_kind)
                    group_kind = str(spec.get("candidate_group_kind"))
                    candidate_group, group_source = group_for_spec(state, fixed_top64, inventory, group_kind)
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
                    row["row_kind"] = "phase933_loso_case_residual_holdout_row"
                    row["phase925_surface_state_key"] = state_key
                    row["candidate_group_kind"] = group_kind
                    row["candidate_group_source"] = group_source
                    row["candidate_group_factor"] = float(spec.get("candidate_group_factor") or 1.0)
                    row["candidate_group_size"] = len(candidate_group)
                    row["phase925_group_kind"] = state["source_row"].get("group_kind")
                    row["phase925_factor"] = state["source_row"].get("factor")
                    rows.append(row)
            if state_idx % max(1, int(args.log_every)) == 0 or state_idx == len(states):
                log(f"{args.model}/{args.round_name}: loso_state={state_idx}/{len(states)} rows={len(rows)}")
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
    comparison = repair_comparison(rows, args)
    label = evidence_label(len(selected), rows)
    payload = {
        "phase": PHASE,
        "title": "LOSO Case Residual Holdout Audit",
        "model": args.model,
        "status": "complete",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "attn_implementation": attn_impl,
        "selected_punctuation_seeds": len(selected),
        "coordinate_count": len(coords),
        "channel_spec_count": len(specs),
        "expected_rows_if_all_reconstructed": len(selected) * len(coords) * len(specs),
        "fixed_group_sizes": {name: len(group) for name, group in fixed_groups.items()},
        "loso_residual_inventory": {k: v for k, v in inventory.items() if k != "by_state"},
        "overall": p928.summarize_rows(rows),
        "by_group_factor": p928.summarize_by(rows, ["candidate_group_kind", "candidate_group_factor"], limit=180),
        "by_group_case_factor": p928.summarize_by(rows, ["candidate_group_kind", "target_case_id", "candidate_group_factor"], limit=260),
        "repair_comparison": comparison,
        "evidence_label": label,
        "boundary": "leave-one-state-out residual holdout only; no natural gate or strict-clean closure claim",
    }
    p846.write_json(out_dir / f"phase933_{args.model}_summary.json", payload)
    p846.write_jsonl(out_dir / f"phase933_{args.model}_rows.jsonl", rows)
    p846.write_jsonl(out_dir / f"phase933_{args.model}_selected_seeds.jsonl", selected)
    print(json.dumps({"phase": PHASE, "model": args.model, "status": label, "overall": payload["overall"], "repair": comparison[:5]}, ensure_ascii=False, indent=2), flush=True)
    return payload


def summarize_round(round_name: str) -> dict[str, Any]:
    out_dir = RESULT_ROOT / round_name
    summaries = [read_json(out_dir / f"phase933_{model}_summary.json") for model in MODELS]
    summaries = [summary for summary in summaries if summary]
    evidence_counts: Counter[str] = Counter()
    overall_scalar: dict[str, Any] = {}
    group_rows: list[dict[str, Any]] = []
    repair_rows: list[dict[str, Any]] = []
    inventories: dict[str, Any] = {}
    for summary in summaries:
        model_name = str(summary.get("model"))
        evidence_counts[str(summary.get("evidence_label"))] += 1
        overall_scalar["selected_punctuation_seeds"] = overall_scalar.get("selected_punctuation_seeds", 0) + int(summary.get("selected_punctuation_seeds") or 0)
        overall_scalar["expected_rows_if_all_reconstructed"] = overall_scalar.get("expected_rows_if_all_reconstructed", 0) + int(summary.get("expected_rows_if_all_reconstructed") or 0)
        for key, value in (summary.get("overall") or {}).items():
            if isinstance(value, int):
                overall_scalar[f"overall_{key}"] = overall_scalar.get(f"overall_{key}", 0) + value
        for row in summary.get("by_group_factor") or []:
            item = dict(row)
            item["model"] = model_name
            group_rows.append(item)
        for row in summary.get("repair_comparison") or []:
            item = dict(row)
            item["model"] = model_name
            repair_rows.append(item)
        inventories[model_name] = summary.get("loso_residual_inventory")
    group_rows.sort(key=lambda row: (int(row.get("target_state_coverage_top1") or 0), int(row.get("new_top1_vs_coordinate_base") or 0), float(row.get("mean_margin_delta_vs_coordinate_base") or -999)), reverse=True)
    repair_rows.sort(key=lambda row: (int(row.get("state_coverage_all") or 0), float(row.get("candidate_group_factor") or 0.0)), reverse=True)
    payload = {
        "phase": PHASE,
        "round": round_name,
        "status": "complete" if summaries else "missing",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "models": [summary.get("model") for summary in summaries],
        "overall_scalar": overall_scalar,
        "evidence_label_counts": dict(evidence_counts),
        "loso_residual_inventories": inventories,
        "top_group_factor_rows": group_rows[:140],
        "top_repair_comparison_rows": repair_rows[:80],
        "model_summaries": summaries,
    }
    p846.write_json(out_dir / "phase933_cross_model_summary.json", payload)
    write_summary_md(out_dir / "phase933_cross_model_summary.md", payload)
    return payload


def write_summary_md(path: Path, payload: dict[str, Any]) -> None:
    lines = ["# Phase 933 LOSO case residual holdout audit", "", "## Overall", ""]
    for key, value in sorted((payload.get("overall_scalar") or {}).items()):
        lines.append(f"- {key}: {value}")
    lines += ["", "## Evidence", ""]
    for key, value in sorted((payload.get("evidence_label_counts") or {}).items()):
        lines.append(f"- {key}: {value}")
    lines += ["", "## LOSO Residual Inventory", ""]
    for model_name, inventory in sorted((payload.get("loso_residual_inventories") or {}).items()):
        lines.append(f"### {model_name}")
        lines.append("")
        for case_id, item in sorted(((inventory or {}).get("by_case_summary") or {}).items()):
            lines.append(
                f"- {case_id}: states={item.get('state_count')}, holdout_each={item.get('holdout_state_count_each')}, inter_residual_size={item.get('inter_residual_size_min')}..{item.get('inter_residual_size_max')}, union_residual_size={item.get('union_residual_size_min')}..{item.get('union_residual_size_max')}"
            )
        lines.append("")
    lines += ["## Top Repair Coverage", ""]
    lines.append("| model | group | factor | all states |")
    lines.append("| --- | --- | ---: | ---: |")
    for row in payload.get("top_repair_comparison_rows") or []:
        lines.append("| {model} | {candidate_group_kind} | {candidate_group_factor} | {state_coverage_all} |".format(**row))
    lines += ["", "## Top Group Factor Rows", ""]
    lines.append("| model | group | factor | rows | top1 | margin | strict | new top1 | states | mean delta |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in payload.get("top_group_factor_rows") or []:
        lines.append(
            "| {model} | {candidate_group_kind} | {candidate_group_factor} | {rows} | {top1} | {margin_nonnegative} | {strict_clean_candidate} | {new_top1_vs_coordinate_base} | {target_state_coverage_top1} | {mean_margin_delta_vs_coordinate_base} |".format(
                **row
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--round-name", default="loso_case_residual_holdout_audit")
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
