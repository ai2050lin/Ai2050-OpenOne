#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

import phase846_geometry_boundary_equation_fitting as p846  # noqa: E402


PHASE = 925
MODELS = ["qwen3", "glm4", "deepseek7b"]
RESULT_ROOT = Path("tests/result/phase925_response_surface_generalization_dataset_expansion")
PHASE914_ROOT = Path("tests/result/phase914_l4_mlp_route_near_holdout_validation")
PHASE915_ROOT = Path("tests/result/phase915_near_boundary_action_gate_search")
PHASE924_ROOT = Path("tests/result/phase924_route_protocol_response_surface_audit")


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


def median(values: list[float | int | None]) -> float | None:
    cleaned = [float(value) for value in values if value is not None]
    return None if not cleaned else float(statistics.median(cleaned))


def mean(values: list[float | int | None]) -> float | None:
    cleaned = [float(value) for value in values if value is not None]
    return None if not cleaned else float(sum(cleaned) / len(cleaned))


def norm_factor(value: Any) -> str:
    if value is None:
        return "None"
    try:
        return f"{float(value):g}"
    except Exception:
        return str(value)


def state_key(row: dict[str, Any]) -> str:
    return "|".join(
        [
            str(row.get("case_id")),
            str(row.get("prompt_variant")),
            str(row.get("source_subset_key")),
            str(row.get("edit_mode")),
            str(row.get("eval_kind")),
            str(row.get("group_kind")),
            norm_factor(row.get("factor")),
        ]
    )


def phase915_state_key(row: dict[str, Any]) -> str:
    return "|".join(
        [
            str(row.get("case_id")),
            str(row.get("prompt_variant")),
            str(row.get("source_subset_key")),
            str(row.get("edit_mode")),
            str(row.get("eval_kind")),
            str(row.get("boundary_group_kind")),
            norm_factor(row.get("boundary_factor")),
        ]
    )


def existing_surface_keys(model_name: str, phase924_round: str) -> set[str]:
    rows_path = PHASE924_ROOT / phase924_round / f"phase924_{model_name}_rows.jsonl"
    keys = set()
    for row in read_jsonl(rows_path):
        key = row.get("target_state_key")
        if key:
            keys.add(str(key))
    return keys


def phase915_keys(model_name: str, phase915_round: str) -> set[str]:
    keys = set()
    for row in read_jsonl(PHASE915_ROOT / phase915_round / f"phase915_{model_name}_rows.jsonl"):
        keys.add(phase915_state_key(row))
    return keys


def margin_value(row: dict[str, Any]) -> float | None:
    value = row.get("patched_eos_margin_vs_blocker")
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def rank_value(row: dict[str, Any]) -> int | None:
    value = row.get("patched_eos_rank")
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def candidate_flags(row: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    margin = margin_value(row)
    rank = rank_value(row)
    top5 = bool(row.get("patched_eos_top5"))
    top10 = bool(row.get("patched_eos_top10"))
    top50 = bool(row.get("patched_eos_top50"))
    weak = bool(row.get("weak_holdout_candidate"))
    strong = bool(row.get("strong_holdout_candidate"))
    blocker_token = row.get("patched_blocker_token")
    blocker_a = str(blocker_token) == str(args.target_blocker_token)
    near_margin = margin is not None and float(args.near_margin_min) <= margin <= float(args.near_margin_max)
    rank_near = rank is not None and rank <= int(args.max_eos_rank)
    usable_boundary = row.get("group_kind") is not None and row.get("factor") is not None
    candidate = bool(usable_boundary and (near_margin or top10 or weak or strong or (top50 and blocker_a and rank_near)))
    return {
        "usable_boundary": usable_boundary,
        "candidate": candidate,
        "near_margin": near_margin,
        "rank_near": rank_near,
        "top5": top5,
        "top10": top10,
        "top50": top50,
        "weak_holdout_candidate": weak,
        "strong_holdout_candidate": strong,
        "blocker_is_target": blocker_a,
    }


def score_row(row: dict[str, Any], flags: dict[str, Any]) -> tuple[float, ...]:
    margin = margin_value(row)
    rank = rank_value(row)
    margin_closeness = 0.0 if margin is None else max(0.0, 4.0 - abs(margin))
    rank_score = 0.0 if rank is None else max(0.0, 64.0 - float(rank)) / 64.0
    band_delta = row.get("band16_mean_logit_delta")
    try:
        band_delta_score = max(0.0, -float(band_delta))
    except Exception:
        band_delta_score = 0.0
    eos_delta = row.get("eos_logit_delta_vs_route")
    try:
        eos_delta_score = float(eos_delta)
    except Exception:
        eos_delta_score = 0.0
    return (
        float(flags.get("strong_holdout_candidate")) * 200.0,
        float(flags.get("weak_holdout_candidate")) * 120.0,
        float(flags.get("top5")) * 70.0,
        float(flags.get("top10")) * 40.0,
        float(flags.get("blocker_is_target")) * 25.0,
        margin_closeness * 10.0,
        rank_score * 10.0,
        band_delta_score * 3.0,
        eos_delta_score,
    )


def score_scalar(score: tuple[float, ...]) -> float:
    return float(sum(score))


def compact_seed(row: dict[str, Any], flags: dict[str, Any], score: tuple[float, ...], existing: set[str], phase915: set[str]) -> dict[str, Any]:
    key = state_key(row)
    return {
        "phase": PHASE,
        "row_kind": "phase925_response_surface_seed",
        "model": row.get("model"),
        "surface_state_key": key,
        "case_id": row.get("case_id"),
        "case_split": row.get("case_split"),
        "eval_domain": row.get("eval_domain"),
        "eval_kind": row.get("eval_kind"),
        "object": row.get("object"),
        "canonical_answer": row.get("canonical_answer"),
        "prompt_variant": row.get("prompt_variant"),
        "edit_mode": row.get("edit_mode"),
        "source_key": row.get("source_key"),
        "source_subset_key": row.get("source_subset_key"),
        "prefix_text": row.get("prefix_text"),
        "group_kind": row.get("group_kind"),
        "factor": row.get("factor"),
        "control_label": row.get("control_label"),
        "subunit_family": row.get("subunit_family"),
        "patched_blocker_token": row.get("patched_blocker_token"),
        "patched_eos_rank": row.get("patched_eos_rank"),
        "patched_eos_margin_vs_blocker": row.get("patched_eos_margin_vs_blocker"),
        "patched_eos_top5": row.get("patched_eos_top5"),
        "patched_eos_top10": row.get("patched_eos_top10"),
        "patched_eos_top50": row.get("patched_eos_top50"),
        "strict_clean_candidate": row.get("strict_clean_candidate"),
        "weak_holdout_candidate": row.get("weak_holdout_candidate"),
        "strong_holdout_candidate": row.get("strong_holdout_candidate"),
        "route_eos_rank": row.get("route_eos_rank"),
        "route_delta_norm": row.get("route_delta_norm"),
        "band16_mean_logit_delta": row.get("band16_mean_logit_delta"),
        "band32_mean_logit_delta": row.get("band32_mean_logit_delta"),
        "eos_logit_delta_vs_route": row.get("eos_logit_delta_vs_route"),
        "score_scalar": score_scalar(score),
        "score_tuple": list(score),
        "already_surface_tested_phase924": key in existing,
        "present_in_phase915_boundary_set": key in phase915,
        "new_surface_seed_vs_phase924": key not in existing,
        **flags,
    }


def dedupe_best(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    buckets: dict[str, tuple[dict[str, Any], dict[str, Any], tuple[float, ...]]] = {}
    for row in rows:
        flags = candidate_flags(row, args)
        if not flags["candidate"]:
            continue
        score = score_row(row, flags)
        key = state_key(row)
        if key not in buckets or score > buckets[key][2]:
            buckets[key] = (row, flags, score)
    return [
        {"_source_row": row, "_flags": flags, "_score": score}
        for row, flags, score in buckets.values()
    ]


def select_diverse(candidates: list[dict[str, Any]], args: argparse.Namespace, existing: set[str], phase915: set[str]) -> list[dict[str, Any]]:
    candidates.sort(
        key=lambda item: (
            item["_score"],
            str(item["_source_row"].get("eval_domain")),
            str(item["_source_row"].get("case_id")),
        ),
        reverse=True,
    )
    selected: list[dict[str, Any]] = []
    case_counts: Counter[str] = Counter()
    domain_counts: Counter[str] = Counter()
    group_counts: Counter[str] = Counter()
    for item in candidates:
        row = item["_source_row"]
        case_id = str(row.get("case_id"))
        domain = str(row.get("eval_domain"))
        group = str(row.get("group_kind"))
        if case_counts[case_id] >= int(args.max_per_case):
            continue
        if domain_counts[domain] >= int(args.max_per_domain):
            continue
        if group_counts[group] >= int(args.max_per_group):
            continue
        selected.append(compact_seed(row, item["_flags"], item["_score"], existing, phase915))
        case_counts[case_id] += 1
        domain_counts[domain] += 1
        group_counts[group] += 1
        if len(selected) >= int(args.max_seeds_per_model):
            break
    if len(selected) < int(args.min_seeds_per_model):
        used = {row["surface_state_key"] for row in selected}
        for item in candidates:
            row = item["_source_row"]
            key = state_key(row)
            if key in used:
                continue
            selected.append(compact_seed(row, item["_flags"], item["_score"], existing, phase915))
            used.add(key)
            if len(selected) >= min(int(args.max_seeds_per_model), int(args.min_seeds_per_model)):
                break
    return selected


def summarize_seed_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "rows": len(rows),
        "new_surface_seed_vs_phase924": sum(1 for row in rows if row.get("new_surface_seed_vs_phase924")),
        "already_surface_tested_phase924": sum(1 for row in rows if row.get("already_surface_tested_phase924")),
        "present_in_phase915_boundary_set": sum(1 for row in rows if row.get("present_in_phase915_boundary_set")),
        "blocker_is_target": sum(1 for row in rows if row.get("blocker_is_target")),
        "top5": sum(1 for row in rows if row.get("top5")),
        "top10": sum(1 for row in rows if row.get("top10")),
        "top50": sum(1 for row in rows if row.get("top50")),
        "weak_holdout_candidate": sum(1 for row in rows if row.get("weak_holdout_candidate")),
        "strong_holdout_candidate": sum(1 for row in rows if row.get("strong_holdout_candidate")),
        "strict_clean_candidate": sum(1 for row in rows if row.get("strict_clean_candidate")),
        "unique_cases": len({row.get("case_id") for row in rows}),
        "unique_domains": len({row.get("eval_domain") for row in rows}),
        "unique_prompt_variants": len({row.get("prompt_variant") for row in rows}),
        "unique_groups": len({row.get("group_kind") for row in rows}),
        "median_margin": median([row.get("patched_eos_margin_vs_blocker") for row in rows]),
        "mean_margin": mean([row.get("patched_eos_margin_vs_blocker") for row in rows]),
        "median_rank": median([row.get("patched_eos_rank") for row in rows]),
        "median_score": median([row.get("score_scalar") for row in rows]),
    }


def counter_rows(rows: list[dict[str, Any]], key: str, limit: int = 40) -> list[dict[str, Any]]:
    counter = Counter(str(row.get(key)) for row in rows)
    return [{"key": item, "count": int(count)} for item, count in counter.most_common(limit)]


def summarize_model(model_name: str, raw_rows: list[dict[str, Any]], candidates: list[dict[str, Any]], selected: list[dict[str, Any]]) -> dict[str, Any]:
    if not raw_rows:
        evidence = "no_phase914_rows"
    elif not candidates:
        evidence = "no_expandable_response_surface_candidates"
    elif summarize_seed_rows(selected)["new_surface_seed_vs_phase924"] >= 24 and summarize_seed_rows(selected)["unique_cases"] >= 4:
        evidence = "expanded_surface_seed_set_ready"
    else:
        evidence = "limited_surface_seed_set"
    return {
        "phase": PHASE,
        "title": "Response Surface Generalization Dataset Expansion",
        "model": model_name,
        "status": "complete",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "phase914_rows": len(raw_rows),
        "candidate_unique_states": len(candidates),
        "selected_surface_seeds": len(selected),
        "raw_summary": {
            "unique_cases": len({row.get("case_id") for row in raw_rows}),
            "unique_domains": len({row.get("eval_domain") for row in raw_rows}),
            "top50": sum(1 for row in raw_rows if row.get("patched_eos_top50")),
            "top10": sum(1 for row in raw_rows if row.get("patched_eos_top10")),
            "top5": sum(1 for row in raw_rows if row.get("patched_eos_top5")),
        },
        "candidate_summary": summarize_seed_rows([
            compact_seed(item["_source_row"], item["_flags"], item["_score"], set(), set()) for item in candidates
        ]),
        "selected_summary": summarize_seed_rows(selected),
        "selected_by_domain": counter_rows(selected, "eval_domain"),
        "selected_by_case": counter_rows(selected, "case_id", 80),
        "selected_by_prompt_variant": counter_rows(selected, "prompt_variant"),
        "selected_by_group": counter_rows(selected, "group_kind"),
        "selected_by_blocker": counter_rows(selected, "patched_blocker_token"),
        "evidence_label": evidence,
        "boundary": (
            "Phase925 does not run new model forwards. It expands the response-surface candidate dataset "
            "from prior Phase914/915/924 outputs and writes selected near-boundary surface seeds for the next "
            "causal surface test."
        ),
    }


def eval_model(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = RESULT_ROOT / args.round_name
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = PHASE914_ROOT / args.phase914_round / f"phase914_{args.model}_rows.jsonl"
    raw_rows = read_jsonl(raw_path)
    existing = existing_surface_keys(args.model, args.phase924_round)
    phase915 = phase915_keys(args.model, args.phase915_round)
    candidates = dedupe_best(raw_rows, args)
    selected = select_diverse(candidates, args, existing, phase915)
    payload = summarize_model(args.model, raw_rows, candidates, selected)
    p846.write_json(out_dir / f"phase925_{args.model}_summary.json", payload)
    p846.write_jsonl(out_dir / f"phase925_{args.model}_selected_surface_seeds.jsonl", selected)
    p846.write_jsonl(
        out_dir / f"phase925_{args.model}_candidate_surface_seeds.jsonl",
        [compact_seed(item["_source_row"], item["_flags"], item["_score"], existing, phase915) for item in candidates],
    )
    print(
        json.dumps(
            {
                "phase": PHASE,
                "model": args.model,
                "phase914_rows": len(raw_rows),
                "candidate_unique_states": len(candidates),
                "selected_surface_seeds": len(selected),
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
    selected = []
    evidence = Counter()
    scalar = Counter()
    for model_name in MODELS:
        summary_path = out_dir / f"phase925_{model_name}_summary.json"
        if not summary_path.exists():
            continue
        summary = read_json(summary_path)
        summaries.append(summary)
        evidence[str(summary.get("evidence_label"))] += 1
        for key in ["phase914_rows", "candidate_unique_states", "selected_surface_seeds"]:
            scalar[key] += int(summary.get(key) or 0)
        for key, value in (summary.get("selected_summary") or {}).items():
            if isinstance(value, int):
                scalar[f"selected_{key}"] += value
        for row in read_jsonl(out_dir / f"phase925_{model_name}_selected_surface_seeds.jsonl"):
            selected.append(row)
    selected.sort(key=lambda row: (row.get("score_scalar") or 0, row.get("patched_eos_top5") or False), reverse=True)
    payload = {
        "phase": PHASE,
        "round": round_name,
        "status": "complete" if len(summaries) == len(MODELS) else "partial",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "models": [summary.get("model") for summary in summaries],
        "overall_scalar": {key: int(value) for key, value in sorted(scalar.items())},
        "evidence_label_counts": dict(sorted(evidence.items())),
        "model_summaries": summaries,
        "top_selected_seeds": selected[:160],
    }
    p846.write_json(out_dir / "phase925_cross_model_summary.json", payload)
    write_markdown(out_dir / "phase925_cross_model_summary.md", payload)
    return payload


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Phase 925 response surface generalization dataset expansion",
        "",
        "## Overall",
        "",
        f"- models: {', '.join(payload.get('models') or [])}",
    ]
    for key, value in (payload.get("overall_scalar") or {}).items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Model Summaries", ""])
    lines.append(
        "| model | phase914 rows | candidate states | selected | new vs P924 | cases | domains | blocker target | top5 | top10 | evidence |"
    )
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
    for summary in payload.get("model_summaries") or []:
        selected = summary.get("selected_summary") or {}
        lines.append(
            "| {model} | {raw} | {cands} | {selected_rows} | {new_rows} | {cases} | {domains} | {target} | {top5} | {top10} | {evidence} |".format(
                model=summary.get("model"),
                raw=summary.get("phase914_rows"),
                cands=summary.get("candidate_unique_states"),
                selected_rows=summary.get("selected_surface_seeds"),
                new_rows=selected.get("new_surface_seed_vs_phase924"),
                cases=selected.get("unique_cases"),
                domains=selected.get("unique_domains"),
                target=selected.get("blocker_is_target"),
                top5=selected.get("top5"),
                top10=selected.get("top10"),
                evidence=summary.get("evidence_label"),
            )
        )
    lines.extend(["", "## Top Selected Seeds", ""])
    lines.append(
        "| model | state | case | domain | prompt | group | factor | blocker | rank | margin | score | new vs P924 |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- | ---: | --- | ---: | ---: | ---: | --- |")
    for row in payload.get("top_selected_seeds") or []:
        row = {
            "model": "",
            "surface_state_key": "",
            "case_id": "",
            "eval_domain": "",
            "prompt_variant": "",
            "group_kind": "",
            "factor": "",
            "patched_blocker_token": "",
            "patched_eos_rank": "",
            "patched_eos_margin_vs_blocker": "",
            "score_scalar": "",
            "new_surface_seed_vs_phase924": "",
            **row,
        }
        lines.append(
            "| {model} | {surface_state_key} | {case_id} | {eval_domain} | {prompt_variant} | {group_kind} | {factor} | {patched_blocker_token} | {patched_eos_rank} | {patched_eos_margin_vs_blocker} | {score_scalar} | {new_surface_seed_vs_phase924} |".format(
                **row
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--round-name", default="response_surface_generalization_dataset_expansion")
    parser.add_argument("--phase914-round", default="l4_mlp_route_near_holdout_validation")
    parser.add_argument("--phase915-round", default="near_boundary_action_gate_search")
    parser.add_argument("--phase924-round", default="route_protocol_response_surface_audit")
    parser.add_argument("--target-blocker-token", default="a")
    parser.add_argument("--near-margin-min", type=float, default=-2.0)
    parser.add_argument("--near-margin-max", type=float, default=0.5)
    parser.add_argument("--max-eos-rank", type=int, default=50)
    parser.add_argument("--max-seeds-per-model", type=int, default=96)
    parser.add_argument("--min-seeds-per-model", type=int, default=36)
    parser.add_argument("--max-per-case", type=int, default=10)
    parser.add_argument("--max-per-domain", type=int, default=40)
    parser.add_argument("--max-per-group", type=int, default=36)
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
