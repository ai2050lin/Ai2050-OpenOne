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


PHASE = 929
MODELS = ["qwen3", "glm4", "deepseek7b"]
RESULT_ROOT = Path("tests/result/phase929_punctuation_margin_gear_holdout_validation")
PHASE925_ROOT = Path("tests/result/phase925_response_surface_generalization_dataset_expansion")
PHASE928_ROOT = Path("tests/result/phase928_punctuation_specific_protocol_gear_search")


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def seed_sort_key(row: dict[str, Any]) -> tuple[str, float, str]:
    return (
        str(row.get("case_id")),
        float(row.get("score_scalar") or 0.0),
        str(row.get("surface_state_key")),
    )


def select_punctuation_seeds(model_name: str, args: argparse.Namespace) -> list[dict[str, Any]]:
    source_name = "selected_surface_seeds" if args.seed_source == "selected" else "candidate_surface_seeds"
    path = PHASE925_ROOT / args.phase925_round / f"phase925_{model_name}_{source_name}.jsonl"
    rows = [
        row
        for row in read_jsonl(path)
        if p926.blocker_class(row.get("patched_blocker_token")) == "punctuation_period"
    ]
    rows.sort(key=seed_sort_key)
    selected: list[dict[str, Any]] = []
    per_case: Counter[str] = Counter()
    for row in rows:
        case_id = str(row.get("case_id"))
        if per_case[case_id] >= int(args.max_per_case):
            continue
        selected.append(dict(row))
        per_case[case_id] += 1
        if len(selected) >= int(args.max_punctuation_seeds):
            break
    return selected


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
                "control_label": f"L{args.target_layer}_margin_support_pos_64_holdout_{factor:g}",
                "control_family": "punctuation_margin_support_holdout",
                "control_class": "candidate_gear",
                "candidate_group_kind": "margin_support_pos_64",
                "candidate_group_factor": float(factor),
            }
        )
    for factor in p928.parse_floats(args.eos_control_factors):
        specs.append(
            {
                "control_label": f"L{args.target_layer}_eos_support_64_control_{factor:g}",
                "control_family": "punctuation_eos_support_control",
                "control_class": "control_gear",
                "candidate_group_kind": "eos_support_64",
                "candidate_group_factor": float(factor),
            }
        )
    for factor in p928.parse_floats(args.blocker_control_factors):
        specs.append(
            {
                "control_label": f"L{args.target_layer}_margin_support_neg_64_control_{factor:g}",
                "control_family": "punctuation_negative_margin_control",
                "control_class": "control_gear",
                "candidate_group_kind": "margin_support_neg_64",
                "candidate_group_factor": float(factor),
            }
        )
    return specs


def phase928_reference(round_name: str, model_name: str) -> tuple[set[str], set[str]]:
    rows_path = PHASE928_ROOT / round_name / f"phase928_{model_name}_rows.jsonl"
    selected_path = PHASE928_ROOT / round_name / f"phase928_{model_name}_selected_seeds.jsonl"
    selected_keys = {str(row.get("surface_state_key")) for row in read_jsonl(selected_path)}
    closure_keys = {
        str(row.get("phase925_surface_state_key"))
        for row in read_jsonl(rows_path)
        if row.get("new_top1_vs_coordinate_base") or row.get("new_margin_closure_vs_coordinate_base")
    }
    return selected_keys, closure_keys


def evidence_label(selected_count: int, rows: list[dict[str, Any]]) -> str:
    if selected_count <= 0:
        return "no_punctuation_period_seeds"
    margin_rows = [r for r in rows if r.get("candidate_group_kind") == "margin_support_pos_64"]
    if any(r.get("new_strict_vs_coordinate_base") for r in margin_rows):
        return "punctuation_margin_gear_strict_holdout_positive"
    unseen_positive = [
        r
        for r in margin_rows
        if (r.get("new_top1_vs_coordinate_base") or r.get("new_margin_closure_vs_coordinate_base"))
        and not r.get("phase928_selected_seed")
    ]
    if unseen_positive:
        return "punctuation_margin_gear_unseen_seed_positive"
    if any(r.get("new_top1_vs_coordinate_base") or r.get("new_margin_closure_vs_coordinate_base") for r in margin_rows):
        return "punctuation_margin_gear_seen_seed_only_positive"
    if any(r.get("improved_margin_vs_coordinate_base") for r in margin_rows):
        return "punctuation_margin_gear_moves_boundary_without_closure"
    return "punctuation_margin_gear_no_effect"


def summarize_model_rows(selected_count: int, rows: list[dict[str, Any]]) -> dict[str, Any]:
    margin_rows = [r for r in rows if r.get("candidate_group_kind") == "margin_support_pos_64"]
    return {
        "selected_punctuation_seeds": selected_count,
        "overall": p928.summarize_rows(rows),
        "margin_group": p928.summarize_rows(margin_rows),
        "by_case": p928.summarize_by(rows, ["target_case_id"], limit=80),
        "by_group_factor": p928.summarize_by(rows, ["candidate_group_kind", "candidate_group_factor"], limit=120),
        "by_seen_status": p928.summarize_by(rows, ["phase928_selected_seed", "phase928_new_closure_seed"], limit=20),
        "new_margin_rows": [
            r
            for r in margin_rows
            if r.get("new_top1_vs_coordinate_base") or r.get("new_margin_closure_vs_coordinate_base")
        ][:120],
    }


def eval_model(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = RESULT_ROOT / args.round_name
    out_dir.mkdir(parents=True, exist_ok=True)
    selected = select_punctuation_seeds(args.model, args)
    coords = p928.parse_coordinate_pairs(args.coordinate_pairs)
    specs = build_specs(args)
    phase928_selected, phase928_closure = phase928_reference(args.phase928_round, args.model)
    if args.dry_run or not selected:
        payload = {
            "phase": PHASE,
            "title": "Punctuation Margin Gear Holdout Validation",
            "model": args.model,
            "status": "dry_run" if args.dry_run else "no_punctuation_period_seeds",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "seed_source": args.seed_source,
            "selected_punctuation_seeds": len(selected),
            "coordinate_count": len(coords),
            "channel_spec_count": len(specs),
            "expected_rows_if_all_reconstructed": len(selected) * len(coords) * len(specs),
            "phase928_selected_reference_count": len(phase928_selected),
            "phase928_closure_reference_count": len(phase928_closure),
            "overall": p928.summarize_rows([]),
            "margin_group": p928.summarize_rows([]),
            "evidence_label": "no_punctuation_period_seeds" if not selected else "dry_run",
        }
        p846.write_json(out_dir / f"phase929_{args.model}_summary.json", payload)
        p846.write_jsonl(out_dir / f"phase929_{args.model}_rows.jsonl", [])
        p846.write_jsonl(out_dir / f"phase929_{args.model}_selected_seeds.jsonl", selected)
        print(json.dumps({"phase": PHASE, "model": args.model, "status": payload["status"], "selected": len(selected)}, ensure_ascii=False, indent=2), flush=True)
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
            log(f"{args.model}/{args.round_name}: reconstructed_holdout_seed={idx}/{len(selected)} kept={len(states)}")
        for state_idx, state in enumerate(states, 1):
            state_key = str(state["source_row"].get("surface_state_key"))
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
                    row["row_kind"] = "phase929_punctuation_margin_holdout_row"
                    row["phase925_surface_state_key"] = state_key
                    row["phase925_seed_blocker_token"] = state["source_row"].get("patched_blocker_token")
                    row["phase925_seed_blocker_class"] = p926.blocker_class(state["source_row"].get("patched_blocker_token"))
                    row["phase925_group_kind"] = state["source_row"].get("group_kind")
                    row["phase925_factor"] = state["source_row"].get("factor")
                    row["phase928_selected_seed"] = state_key in phase928_selected
                    row["phase928_new_closure_seed"] = state_key in phase928_closure
                    row["candidate_group_kind"] = group_kind
                    row["candidate_group_factor"] = float(spec.get("candidate_group_factor") or 1.0)
                    row["candidate_group_size"] = len(candidate_group)
                    rows.append(row)
            if state_idx % max(1, int(args.log_every)) == 0 or state_idx == len(states):
                log(f"{args.model}/{args.round_name}: holdout_state={state_idx}/{len(states)} rows={len(rows)}")
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
    model_summary = summarize_model_rows(len(selected), rows)
    label = evidence_label(len(selected), rows)
    payload = {
        "phase": PHASE,
        "title": "Punctuation Margin Gear Holdout Validation",
        "model": args.model,
        "status": "complete",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "attn_implementation": attn_impl,
        "seed_source": args.seed_source,
        "coordinate_count": len(coords),
        "channel_spec_count": len(specs),
        "expected_rows_if_all_reconstructed": len(selected) * len(coords) * len(specs),
        "phase928_selected_reference_count": len(phase928_selected),
        "phase928_closure_reference_count": len(phase928_closure),
        **model_summary,
        "evidence_label": label,
        "boundary": "candidate margin-support gear rule validation only; not natural gate closure",
    }
    p846.write_json(out_dir / f"phase929_{args.model}_summary.json", payload)
    p846.write_jsonl(out_dir / f"phase929_{args.model}_rows.jsonl", rows)
    p846.write_jsonl(out_dir / f"phase929_{args.model}_selected_seeds.jsonl", selected)
    print(json.dumps({"phase": PHASE, "model": args.model, "status": label, "overall": payload["overall"]}, ensure_ascii=False, indent=2), flush=True)
    return payload


def summarize_round(round_name: str) -> dict[str, Any]:
    out_dir = RESULT_ROOT / round_name
    summaries = [read_json(out_dir / f"phase929_{model}_summary.json") for model in MODELS]
    summaries = [summary for summary in summaries if summary]
    overall_scalar: dict[str, Any] = {}
    evidence_counts: Counter[str] = Counter()
    top_rows: list[dict[str, Any]] = []
    for summary in summaries:
        evidence_counts[str(summary.get("evidence_label"))] += 1
        overall_scalar["selected_punctuation_seeds"] = overall_scalar.get("selected_punctuation_seeds", 0) + int(summary.get("selected_punctuation_seeds") or 0)
        overall_scalar["expected_rows_if_all_reconstructed"] = overall_scalar.get("expected_rows_if_all_reconstructed", 0) + int(summary.get("expected_rows_if_all_reconstructed") or 0)
        for key, value in (summary.get("overall") or {}).items():
            if isinstance(value, int):
                overall_scalar[f"overall_{key}"] = overall_scalar.get(f"overall_{key}", 0) + value
        for row in summary.get("new_margin_rows") or []:
            item = dict(row)
            item["model"] = summary.get("model")
            top_rows.append(item)
    top_rows.sort(key=lambda r: (float(r.get("margin_delta_vs_coordinate_base") or 0.0), float(r.get("patched_eos_margin_vs_blocker") or -999.0)), reverse=True)
    payload = {
        "phase": PHASE,
        "round": round_name,
        "status": "complete" if summaries else "missing",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "models": [summary.get("model") for summary in summaries],
        "overall_scalar": overall_scalar,
        "evidence_label_counts": dict(evidence_counts),
        "top_new_margin_rows": top_rows[:80],
        "model_summaries": summaries,
    }
    p846.write_json(out_dir / "phase929_cross_model_summary.json", payload)
    write_summary_md(out_dir / "phase929_cross_model_summary.md", payload)
    return payload


def write_summary_md(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Phase 929 punctuation margin gear holdout validation",
        "",
        "## Overall",
        "",
    ]
    for key, value in sorted((payload.get("overall_scalar") or {}).items()):
        lines.append(f"- {key}: {value}")
    lines += ["", "## Evidence", ""]
    for key, value in sorted((payload.get("evidence_label_counts") or {}).items()):
        lines.append(f"- {key}: {value}")
    lines += ["", "## Top New Margin Rows", ""]
    lines.append("| model | state | case | seen | closure_seen | group | factor | alpha | protocol | rank | margin | base margin | delta | strict |")
    lines.append("| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
    for row in payload.get("top_new_margin_rows") or []:
        lines.append(
            "| {model} | {target_state_key} | {target_case_id} | {phase928_selected_seed} | {phase928_new_closure_seed} | {candidate_group_kind} | {candidate_group_factor} | {route_alpha} | {protocol_span_factor} | {patched_eos_rank} | {patched_eos_margin_vs_blocker} | {coordinate_base_margin} | {margin_delta_vs_coordinate_base} | {strict_clean_candidate} |".format(
                **row
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--round-name", default="punctuation_margin_gear_holdout_validation")
    parser.add_argument("--phase925-round", default="response_surface_generalization_dataset_expansion")
    parser.add_argument("--phase928-round", default="punctuation_specific_protocol_gear_search")
    parser.add_argument("--seed-source", choices=["selected", "candidate"], default="selected")
    parser.add_argument("--max-punctuation-seeds", type=int, default=30)
    parser.add_argument("--max-per-case", type=int, default=10)
    parser.add_argument("--coordinate-pairs", default="1.0:1.0,0.875:1.1,1.25:1.1,0.875:0.85,1.25:0.85,1.375:0.85,1.375:0.9")
    parser.add_argument("--margin-factors", default="1.25,1.5,1.75,2.0,2.25")
    parser.add_argument("--eos-control-factors", default="2.0")
    parser.add_argument("--blocker-control-factors", default="0.25")
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
