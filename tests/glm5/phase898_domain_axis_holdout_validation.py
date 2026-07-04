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
import phase856_identity_class_overlap_cross_domain_rollout_audit as p856  # noqa: E402
import phase862_negative_blocker_sign_mechanism_audit as p862  # noqa: E402
import phase885_stable_boundary_minimality_cross_model_audit as p885  # noqa: E402
import phase895_no_single_minimality_head_pathway_split as p895  # noqa: E402


PHASE = 898
MODELS = ["qwen3", "glm4", "deepseek7b"]
RESULT_ROOT = Path("tests/result/phase898_domain_axis_holdout_validation")
PHASE897_ROOT = Path("tests/result/phase897_non_color_route_axis_discovery")
PHASE897_ROUND = "non_color_axis_discovery"


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def parse_csv(text: str) -> list[str]:
    return [part.strip() for part in str(text or "").split(",") if part.strip()]


def finite(value: Any, default: float = 0.0) -> float:
    return p846.finite(value, default)


def mean(values: list[float]) -> float | None:
    return p846.mean(values)


def counter_values(counter: Counter[str]) -> dict[str, int]:
    return {key: int(value) for key, value in sorted(counter.items())}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def gear_key(gear: dict[str, Any]) -> str:
    return p862.gear_key(gear)


def parse_gear_key(text: str) -> dict[str, Any] | None:
    return p862.parse_gear_key(str(text))


def gear_keys_from_subset(subset_key: str) -> list[str]:
    return [part for part in str(subset_key or "").split("+") if part.startswith("L") and "C" in part]


def source_key(source: dict[str, Any]) -> str:
    return f"{source['source_type']}::{source['domain']}::{source['subset_key']}"


def load_phase897_sources(model_name: str, args: argparse.Namespace) -> list[dict[str, Any]]:
    summary = read_json(PHASE897_ROOT / args.phase897_round / f"phase897_{model_name}_summary.json")
    sources: list[dict[str, Any]] = []
    for row in summary.get("pair_domain_groups") or []:
        if int(row.get("known_axis_minimal_pair_closure") or 0) > 0 or int(row.get("no_single_pair_closure") or 0) > 0:
            sources.append(
                {
                    "model": model_name,
                    "source_type": "phase897_pair_candidate",
                    "domain": row.get("domain"),
                    "subset_key": row.get("subset_key"),
                    "subset_size": 2,
                    "phase897_closure_from_open": int(row.get("closure_from_open") or 0),
                    "phase897_no_single_pair": int(row.get("no_single_pair_closure") or 0),
                    "phase897_known_minimal": int(row.get("known_axis_minimal_pair_closure") or 0),
                    "phase897_mean_target_lift": row.get("mean_target_lift"),
                    "phase897_mean_blocker_reduction": row.get("mean_blocker_reduction"),
                }
            )
    singles = [
        row
        for row in summary.get("single_domain_groups") or []
        if int(row.get("closure_from_open") or 0) >= int(args.min_single_closure)
    ]
    singles.sort(key=lambda row: (int(row.get("closure_from_open") or 0), finite(row.get("mean_target_lift"))), reverse=True)
    for row in singles[: int(args.max_single_candidates_per_model)]:
        sources.append(
            {
                "model": model_name,
                "source_type": "phase897_single_candidate",
                "domain": row.get("domain"),
                "subset_key": row.get("subset_key"),
                "subset_size": 1,
                "phase897_closure_from_open": int(row.get("closure_from_open") or 0),
                "phase897_no_single_pair": 0,
                "phase897_known_minimal": 0,
                "phase897_mean_target_lift": row.get("mean_target_lift"),
                "phase897_mean_blocker_reduction": row.get("mean_blocker_reduction"),
            }
        )
    sources.sort(
        key=lambda row: (
            row["source_type"] == "phase897_pair_candidate",
            int(row.get("phase897_known_minimal") or 0),
            int(row.get("phase897_no_single_pair") or 0),
            int(row.get("phase897_closure_from_open") or 0),
        ),
        reverse=True,
    )
    seen: set[str] = set()
    out = []
    for source in sources:
        key = source_key(source)
        if key in seen:
            continue
        seen.add(key)
        out.append(source)
        if len(out) >= int(args.max_sources_per_model):
            break
    return out


def selected_conditions(source: dict[str, Any], args: argparse.Namespace) -> list[dict[str, Any]]:
    domain = str(source.get("domain"))
    max_per_domain = int(args.max_cases_per_domain)
    rows = [dict(case) for case in p885.extended_cases() if str(case.get("domain")) == domain]
    rows.sort(key=lambda case: (str(case.get("split_source", "phase856_base")), str(case.get("object"))))
    if max_per_domain > 0:
        rows = rows[:max_per_domain]
    out: list[dict[str, Any]] = []
    for case in rows:
        for prompt_variant in parse_csv(args.prompt_variants):
            for edit_mode in parse_csv(args.edit_modes):
                item = dict(case)
                item["prompt_variant"] = prompt_variant
                item["edit_mode"] = edit_mode
                item["case_split"] = case.get("split_source", "phase856_base")
                item["source_key"] = source_key(source)
                out.append(item)
    return out


def specs_for_source(source: dict[str, Any]) -> list[dict[str, Any]]:
    keys = gear_keys_from_subset(str(source.get("subset_key")))
    gears = [parse_gear_key(key) for key in keys]
    gears = [gear for gear in gears if gear is not None]
    if not gears:
        return []
    specs = []
    if int(source.get("subset_size") or len(gears)) == 2 and len(gears) == 2:
        for gear in gears:
            specs.append(
                {
                    "subset_key": gear_key(gear),
                    "subset_size": 1,
                    "subset_relation": "component_single",
                    "gear_keys": [gear_key(gear)],
                    "gears": [gear],
                }
            )
        specs.append(
            {
                "subset_key": "+".join(gear_key(gear) for gear in gears),
                "subset_size": 2,
                "subset_relation": "phase897_pair",
                "gear_keys": [gear_key(gear) for gear in gears],
                "gears": gears,
            }
        )
    else:
        gear = gears[0]
        specs.append(
            {
                "subset_key": gear_key(gear),
                "subset_size": 1,
                "subset_relation": "phase897_single",
                "gear_keys": [gear_key(gear)],
                "gears": [gear],
            }
        )
    return specs


def make_row(
    model_name: str,
    source: dict[str, Any],
    condition: dict[str, Any],
    spec: dict[str, Any],
    base_metrics: dict[str, Any],
    metrics: dict[str, Any],
) -> dict[str, Any]:
    return {
        "phase": PHASE,
        "row_kind": "phase898_holdout_row",
        "model": model_name,
        "source_key": source_key(source),
        "source_type": source.get("source_type"),
        "source_subset_key": source.get("subset_key"),
        "eval_domain": source.get("domain"),
        "case_id": condition.get("case_id"),
        "case_split": condition.get("case_split"),
        "object": condition.get("object"),
        "prompt_variant": condition.get("prompt_variant"),
        "edit_mode": condition.get("edit_mode"),
        "subset_key": spec.get("subset_key"),
        "subset_size": spec.get("subset_size"),
        "subset_relation": spec.get("subset_relation"),
        "gear_keys": spec.get("gear_keys"),
        "base_boundary_closed": bool(base_metrics.get("class_boundary_closed")),
        "boundary_closed": bool(metrics.get("class_boundary_closed")),
        "closure_from_open": bool((not base_metrics.get("class_boundary_closed")) and metrics.get("class_boundary_closed")),
        "target_lift": p895.target_lift(base_metrics, metrics),
        "base_class_rank": base_metrics.get("class_best_rank"),
        "class_rank": metrics.get("class_best_rank"),
        "base_full_class_blocker_count": base_metrics.get("full_class_blocker_count"),
        "full_class_blocker_count": metrics.get("full_class_blocker_count"),
        "full_blocker_reduction": p895.blocker_reduction(base_metrics, metrics),
        "full_top_blocker_token": metrics.get("full_class_top_blocker_token"),
        "full_top_blocker_role": metrics.get("full_class_top_blocker_role"),
        "class_minus_object_logit": metrics.get("full_class_minus_object_logit"),
    }


def add_condition_fields(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            str(row.get("model")),
            str(row.get("source_key")),
            str(row.get("case_id")),
            str(row.get("prompt_variant")),
            str(row.get("edit_mode")),
        )
        groups[key].append(row)
    out = []
    for key, vals in groups.items():
        singles = [row for row in vals if int(row.get("subset_size") or 0) == 1]
        pairs = [row for row in vals if int(row.get("subset_size") or 0) == 2]
        single_closure = {str(row.get("subset_key")): bool(row.get("closure_from_open")) for row in singles}
        pair_closure_keys = [str(row.get("subset_key")) for row in pairs if row.get("closure_from_open")]
        no_single_pair_keys = []
        for row in pairs:
            components = [str(item) for item in row.get("gear_keys") or []]
            row["any_component_single_closure"] = any(single_closure.get(item, False) for item in components)
            row["no_single_pair_closure"] = bool(row.get("closure_from_open") and not row["any_component_single_closure"])
            if row["no_single_pair_closure"]:
                no_single_pair_keys.append(str(row.get("subset_key")))
        out.append(
            {
                "phase": PHASE,
                "row_kind": "phase898_condition_summary",
                "model": key[0],
                "source_key": key[1],
                "case_id": key[2],
                "prompt_variant": key[3],
                "edit_mode": key[4],
                "eval_domain": vals[0].get("eval_domain") if vals else None,
                "object": vals[0].get("object") if vals else None,
                "source_type": vals[0].get("source_type") if vals else None,
                "source_subset_key": vals[0].get("source_subset_key") if vals else None,
                "any_single_axis_closure": any(single_closure.values()),
                "single_closure_keys": sorted([name for name, closed in single_closure.items() if closed]),
                "pair_closure_keys": sorted(pair_closure_keys),
                "no_single_pair_keys": sorted(no_single_pair_keys),
                "source_candidate_closure": any(
                    row.get("closure_from_open") and str(row.get("subset_key")) == str(vals[0].get("source_subset_key"))
                    for row in vals
                ),
            }
        )
    return out


def summarize_model(
    model_name: str,
    sources: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    condition_rows: list[dict[str, Any]],
    attn_impl: str | None,
) -> dict[str, Any]:
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in condition_rows:
        by_source[str(row.get("source_key"))].append(row)
    source_summaries = []
    for source in sources:
        key = source_key(source)
        vals = by_source.get(key, [])
        source_summaries.append(
            {
                **source,
                "source_key": key,
                "conditions": len(vals),
                "source_candidate_closure": sum(1 for row in vals if row.get("source_candidate_closure")),
                "single_axis_closure_conditions": sum(1 for row in vals if row.get("any_single_axis_closure")),
                "pair_closure_conditions": sum(1 for row in vals if row.get("pair_closure_keys")),
                "no_single_pair_conditions": sum(1 for row in vals if row.get("no_single_pair_keys")),
                "single_keys": counter_values(Counter(key2 for row in vals for key2 in (row.get("single_closure_keys") or []))),
                "pair_keys": counter_values(Counter(key2 for row in vals for key2 in (row.get("no_single_pair_keys") or []))),
            }
        )
    source_summaries.sort(
        key=lambda row: (
            row.get("no_single_pair_conditions") or 0,
            row.get("source_candidate_closure") or 0,
            row.get("single_axis_closure_conditions") or 0,
        ),
        reverse=True,
    )
    overall = {
        "sources": len(sources),
        "rows": len(rows),
        "condition_rows": len(condition_rows),
        "source_candidate_closure_conditions": sum(1 for row in condition_rows if row.get("source_candidate_closure")),
        "single_axis_closure_conditions": sum(1 for row in condition_rows if row.get("any_single_axis_closure")),
        "pair_closure_conditions": sum(1 for row in condition_rows if row.get("pair_closure_keys")),
        "no_single_pair_conditions": sum(1 for row in condition_rows if row.get("no_single_pair_keys")),
    }
    if overall["no_single_pair_conditions"]:
        evidence_label = "holdout_no_single_pair_supported"
    elif overall["source_candidate_closure_conditions"]:
        evidence_label = "holdout_candidate_closure_without_minimality"
    elif overall["single_axis_closure_conditions"]:
        evidence_label = "holdout_single_axis_supported"
    else:
        evidence_label = "holdout_candidate_not_stable"
    return {
        "phase": PHASE,
        "title": "Domain Axis Holdout Validation",
        "model": model_name,
        "status": "complete",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "attn_implementation": attn_impl,
        "sources": sources,
        "overall": overall,
        "source_summaries": source_summaries,
        "evidence_label": evidence_label,
        "boundary": (
            "Phase898 re-tests Phase897 domain axes on expanded objects and prompts. "
            "It validates candidate stability, not full language mechanism closure."
        ),
    }


def eval_model(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = RESULT_ROOT / args.round_name
    out_dir.mkdir(parents=True, exist_ok=True)
    sources = load_phase897_sources(args.model, args)
    if args.dry_run or not sources:
        payload = {
            "phase": PHASE,
            "model": args.model,
            "round": args.round_name,
            "status": "dry_run" if sources else "no_sources",
            "sources": sources,
        }
        p846.write_json(out_dir / f"phase898_{args.model}_summary.json", payload)
        p846.write_jsonl(out_dir / f"phase898_{args.model}_rows.jsonl", [])
        p846.write_jsonl(out_dir / f"phase898_{args.model}_condition_rows.jsonl", [])
        print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
        return payload

    model = None
    tokenizer = None
    rows: list[dict[str, Any]] = []
    condition_rows: list[dict[str, Any]] = []
    attn_impl = None
    try:
        model, tokenizer, device, attn_impl = p862.p844.p828.p796.load_model_bf16_prefer_flash(
            args.model, args.attn_implementations
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        base_cache: dict[tuple[str, str], tuple[dict[str, Any], dict[str, Any], list[int]]] = {}
        for sidx, source in enumerate(sources, 1):
            conditions = selected_conditions(source, args)
            specs = specs_for_source(source)
            for idx, condition in enumerate(conditions, 1):
                base_key = (str(condition.get("case_id")), str(condition.get("prompt_variant")))
                if base_key not in base_cache:
                    prompt = p885.prompt_for_case(condition, str(condition.get("prompt_variant")))
                    prompt_ids = p862.p844.encode_prompt(tokenizer, prompt)
                    token_sets = p856.token_sets(tokenizer, condition)
                    base_logits = p862.first_logits_with_scaled_gears(model, device, prompt_ids, [], "original", float(args.scale_up_factor))
                    base_cache[base_key] = (
                        p895.metrics_for_logits(tokenizer, base_logits, token_sets, int(args.topk_tokens), int(args.topk_blockers)),
                        token_sets,
                        prompt_ids,
                    )
                base_metrics, token_sets, prompt_ids = base_cache[base_key]
                for spec in specs:
                    logits = p862.first_logits_with_scaled_gears(
                        model,
                        device,
                        prompt_ids,
                        spec["gears"],
                        str(condition.get("edit_mode")),
                        float(args.scale_up_factor),
                    )
                    metrics = p895.metrics_for_logits(tokenizer, logits, token_sets, int(args.topk_tokens), int(args.topk_blockers))
                    rows.append(make_row(args.model, source, condition, spec, base_metrics, metrics))
                if idx % max(1, int(args.log_every)) == 0 or idx == len(conditions):
                    log(f"{args.model}/{args.round_name}: source={sidx}/{len(sources)} condition={idx}/{len(conditions)} rows={len(rows)}")
        condition_rows = add_condition_fields(rows)
    finally:
        if model is not None:
            p862.p844.p828.release_model(model)
        if tokenizer is not None:
            del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    payload = summarize_model(args.model, sources, rows, condition_rows, attn_impl)
    p846.write_json(out_dir / f"phase898_{args.model}_summary.json", payload)
    p846.write_jsonl(out_dir / f"phase898_{args.model}_rows.jsonl", rows)
    p846.write_jsonl(out_dir / f"phase898_{args.model}_condition_rows.jsonl", condition_rows)
    print(json.dumps({"phase": PHASE, "model": args.model, "overall": payload["overall"]}, ensure_ascii=False, indent=2), flush=True)
    return payload


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Phase 898 domain axis holdout validation",
        "",
        "## Overall",
        "",
        f"- models: {', '.join(payload.get('models') or [])}",
    ]
    for key, value in (payload.get("overall") or {}).items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Source summaries", ""])
    lines.append(
        "| model | source | domain | subset | conditions | source closure | single closure | pair closure | no-single | single keys | pair keys |"
    )
    lines.append("| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |")
    for row in payload.get("source_summaries") or []:
        lines.append(
            "| {model} | {source_type} | {domain} | {subset_key} | {conditions} | {source_candidate_closure} | "
            "{single_axis_closure_conditions} | {pair_closure_conditions} | {no_single_pair_conditions} | "
            "{single_keys} | {pair_keys} |".format(**row)
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def summarize_round(round_name: str) -> dict[str, Any]:
    out_dir = RESULT_ROOT / round_name
    summaries = []
    for model_name in MODELS:
        path = out_dir / f"phase898_{model_name}_summary.json"
        if path.exists():
            summaries.append(read_json(path))
    overall: Counter[str] = Counter()
    source_summaries = []
    for summary in summaries:
        for key, value in (summary.get("overall") or {}).items():
            if isinstance(value, int):
                overall[key] += value
        source_summaries.extend(summary.get("source_summaries") or [])
    source_summaries.sort(
        key=lambda row: (
            row.get("no_single_pair_conditions") or 0,
            row.get("source_candidate_closure") or 0,
            row.get("single_axis_closure_conditions") or 0,
        ),
        reverse=True,
    )
    payload = {
        "phase": PHASE,
        "round": round_name,
        "status": "complete" if len(summaries) == len(MODELS) else "partial",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "models": [summary.get("model") for summary in summaries],
        "overall": {key: int(value) for key, value in sorted(overall.items())},
        "source_summaries": source_summaries,
        "evidence_label_counts": counter_values(Counter(str(summary.get("evidence_label")) for summary in summaries)),
    }
    p846.write_json(out_dir / "phase898_cross_model_summary.json", payload)
    write_markdown(out_dir / "phase898_cross_model_summary.md", payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--round-name", default="domain_axis_holdout_validation")
    parser.add_argument("--phase897-round", default=PHASE897_ROUND)
    parser.add_argument("--prompt-variants", default="natural_question,natural_category,classification,object_only")
    parser.add_argument("--edit-modes", default="flip,zero")
    parser.add_argument("--max-cases-per-domain", type=int, default=24)
    parser.add_argument("--max-sources-per-model", type=int, default=8)
    parser.add_argument("--max-single-candidates-per-model", type=int, default=4)
    parser.add_argument("--min-single-closure", type=int, default=4)
    parser.add_argument("--scale-up-factor", type=float, default=2.0)
    parser.add_argument("--topk-tokens", type=int, default=30)
    parser.add_argument("--topk-blockers", type=int, default=50)
    parser.add_argument("--log-every", type=int, default=48)
    parser.add_argument("--attn-implementations", default="flash_attention_2,sdpa")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summarize-round", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.summarize_round:
        payload = summarize_round(args.round_name)
        print(json.dumps({"phase": PHASE, "status": payload["status"], "overall": payload["overall"]}, ensure_ascii=False, indent=2))
        return
    if not args.model:
        raise SystemExit("--model is required unless --summarize-round is set")
    eval_model(args)


if __name__ == "__main__":
    main()
