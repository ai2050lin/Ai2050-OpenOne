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
import phase876_nonclean_route_causal_validation as p876  # noqa: E402


PHASE = 881
MODELS = p846.MODELS
RESULT_ROOT = Path("tests/result/phase881_dominant_gear_robustness_and_repair")
PHASE880_ROWS = Path(
    "tests/result/phase880_counterfactual_gear_min_cut_validation/gear_subset_phase879/phase880_deepseek7b_rows.jsonl"
)
PHASE876_ROOT = Path("tests/result/phase876_nonclean_route_causal_validation/validation")


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def finite(value: Any, default: float = 0.0) -> float:
    return p846.finite(value, default)


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def parse_csv(text: str) -> list[str]:
    return [part.strip() for part in str(text or "").split(",") if part.strip()]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return p846.read_jsonl(path) if path.exists() else []


def gear_key(gear: dict[str, Any]) -> str:
    return p862.gear_key(gear)


def parse_gear_key(key: str) -> dict[str, Any] | None:
    return p862.parse_gear_key(key)


def answer_like(row: dict[str, Any]) -> bool:
    return bool(row.get("rollout_answer_class") or row.get("rollout_clear_answer_class") or row.get("rollout_strict_canonical"))


def boundary_closed(row: dict[str, Any]) -> bool:
    return row.get("class_blocker_count") == 0 and row.get("class_best_rank") == 1


def selected_cases_for_model(model_name: str, max_cases_per_domain: int) -> list[dict[str, Any]]:
    cases = p876.validation_cases()
    if model_name == "qwen3":
        cases = [case for case in cases if str(case.get("domain")) == "material"]
    if int(max_cases_per_domain) <= 0:
        return cases
    counts: Counter[str] = Counter()
    out: list[dict[str, Any]] = []
    for case in cases:
        domain = str(case.get("domain"))
        if counts[domain] >= int(max_cases_per_domain):
            continue
        out.append(case)
        counts[domain] += 1
    return out


def phase880_dominant_candidates(min_hits: int, modes: list[str]) -> list[dict[str, Any]]:
    rows = read_jsonl(PHASE880_ROWS)
    counts: Counter[str] = Counter()
    routes: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        if str(row.get("transition_class")) != "nonclean_output_transition":
            continue
        for key in row.get("proper_subset_boundary_closed_keys") or []:
            if str(key).startswith("subset_"):
                gear = str(key).removeprefix("subset_")
                counts[gear] += 1
                routes[gear][str(row.get("primary_route"))] += 1
    candidates: list[dict[str, Any]] = []
    for key, count in counts.items():
        if count < int(min_hits):
            continue
        gear = parse_gear_key(key)
        if gear is None:
            continue
        for mode in modes:
            candidates.append(
                {
                    "candidate_source": "phase880_dominant_nonclean_subset",
                    "candidate_key": f"{key}:{mode}",
                    "gear": gear,
                    "gear_key": key,
                    "edit_mode": mode,
                    "source_hit_count": count,
                    "source_route_counts": dict(routes[key]),
                }
            )
    return candidates


def qwen3_repair_candidates(modes: list[str]) -> list[dict[str, Any]]:
    path = PHASE876_ROOT / "phase876_qwen3_rows.jsonl"
    gear_keys: set[str] = set()
    for row in read_jsonl(path):
        if row.get("condition_type") == "original":
            continue
        for key in row.get("gear_keys") or []:
            gear_keys.add(str(key))
    candidates: list[dict[str, Any]] = []
    for key in sorted(gear_keys):
        gear = parse_gear_key(key)
        if gear is None:
            continue
        for mode in modes:
            candidates.append(
                {
                    "candidate_source": "phase876_qwen3_single_gear_repair",
                    "candidate_key": f"{key}:{mode}",
                    "gear": gear,
                    "gear_key": key,
                    "edit_mode": mode,
                    "source_hit_count": None,
                    "source_route_counts": {},
                }
            )
    return candidates


def selected_candidates(model_name: str, modes: list[str], min_dominant_hits: int) -> list[dict[str, Any]]:
    if model_name == "deepseek7b":
        return phase880_dominant_candidates(min_dominant_hits, modes)
    if model_name == "qwen3":
        return qwen3_repair_candidates(modes)
    return []


def top_token_compact(tokens: list[dict[str, Any]], limit: int = 8) -> list[dict[str, Any]]:
    return [
        {
            "token_id": item.get("token_id"),
            "token": item.get("token"),
            "role": item.get("role"),
            "logit": item.get("logit"),
            "gap_vs_threshold": item.get("gap_vs_threshold"),
        }
        for item in (tokens or [])[: int(limit)]
    ]


def eval_condition(
    model,
    tokenizer,
    device: torch.device,
    case: dict[str, Any],
    prompt_variant: str,
    gear: dict[str, Any] | None,
    mode: str,
    scale_up_factor: float,
    max_new_tokens: int,
    topk_tokens: int,
    topk_blockers: int,
    original_logits: torch.Tensor | None,
    original_blockers: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    prompt = p876.validation_prompt(case, prompt_variant)
    prompt_ids = p862.p844.encode_prompt(tokenizer, prompt)
    sets = p856.token_sets(tokenizer, case)
    gears = [] if gear is None else [gear]
    effective_mode = "original" if gear is None else mode
    logits = p862.first_logits_with_scaled_gears(model, device, prompt_ids, gears, effective_mode, scale_up_factor)
    first = p856.first_token_metrics(tokenizer, logits, sets, topk_tokens)
    blocker = p862.p854.blocker_metrics(tokenizer, logits, sets, topk_blockers)
    generated, token_ids = p862.greedy_with_scaled_gears(
        model,
        tokenizer,
        device,
        prompt_ids,
        gears,
        effective_mode,
        max_new_tokens,
        scale_up_factor,
    )
    rollout = p856.classify_rollout(generated, case)
    deltas = {}
    if original_logits is not None and original_blockers is not None:
        deltas = p862.original_blocker_deltas(logits, original_logits, original_blockers, topk_blockers)
    row = {
        "prompt": prompt,
        "token_ids": token_ids,
        "generated_clean": p856.clean_text(generated),
        **first,
        **{f"blocker_{key}": value for key, value in blocker.items()},
        **rollout,
        **deltas,
    }
    row["class_boundary_closed"] = boundary_closed(row)
    row["answer_like"] = answer_like(row)
    row["class_top_blockers_compact"] = top_token_compact(row.get("blocker_class_top_blockers") or [])
    row["top_tokens_compact"] = top_token_compact(row.get("blocker_top_tokens") or row.get("top_tokens") or [])
    return row


def make_result_row(
    model_name: str,
    candidate: dict[str, Any],
    case: dict[str, Any],
    prompt_variant: str,
    base: dict[str, Any],
    intervened: dict[str, Any],
) -> dict[str, Any]:
    closure_from_open = bool((not base.get("class_boundary_closed")) and intervened.get("class_boundary_closed"))
    answer_gain = bool((not base.get("answer_like")) and intervened.get("answer_like"))
    original_blocker_delta = intervened.get("original_blocker_delta_mean")
    clean_like = bool(closure_from_open and original_blocker_delta is not None and finite(original_blocker_delta) < 0)
    nonclean_like = bool(closure_from_open and original_blocker_delta is not None and finite(original_blocker_delta) >= 0)
    return {
        "phase": PHASE,
        "row_kind": "phase881_dominant_gear_robustness_row",
        "model": model_name,
        "candidate_source": candidate.get("candidate_source"),
        "candidate_key": candidate.get("candidate_key"),
        "gear_key": candidate.get("gear_key"),
        "edit_mode": candidate.get("edit_mode"),
        "source_hit_count": candidate.get("source_hit_count"),
        "source_route_counts": candidate.get("source_route_counts"),
        "domain": case.get("domain"),
        "case_id": case.get("case_id"),
        "object": case.get("object"),
        "prompt_variant": prompt_variant,
        "base_generated_clean": base.get("generated_clean"),
        "intervened_generated_clean": intervened.get("generated_clean"),
        "base_rollout_label": base.get("rollout_label"),
        "intervened_rollout_label": intervened.get("rollout_label"),
        "base_class_boundary_closed": base.get("class_boundary_closed"),
        "intervened_class_boundary_closed": intervened.get("class_boundary_closed"),
        "closure_from_open": closure_from_open,
        "answer_gain": answer_gain,
        "clean_like_closure": clean_like,
        "nonclean_like_closure": nonclean_like,
        "base_answer_like": base.get("answer_like"),
        "intervened_answer_like": intervened.get("answer_like"),
        "base_class_blocker_count": base.get("class_blocker_count"),
        "intervened_class_blocker_count": intervened.get("class_blocker_count"),
        "base_class_rank": base.get("class_best_rank"),
        "intervened_class_rank": intervened.get("class_best_rank"),
        "base_class_logit": base.get("class_best_logit"),
        "intervened_class_logit": intervened.get("class_best_logit"),
        "class_logit_delta": None
        if base.get("class_best_logit") is None or intervened.get("class_best_logit") is None
        else finite(intervened.get("class_best_logit")) - finite(base.get("class_best_logit")),
        "blocker_reduction": None
        if base.get("class_blocker_count") is None or intervened.get("class_blocker_count") is None
        else finite(base.get("class_blocker_count")) - finite(intervened.get("class_blocker_count")),
        "rank_improvement": None
        if base.get("class_best_rank") is None or intervened.get("class_best_rank") is None
        else finite(base.get("class_best_rank")) - finite(intervened.get("class_best_rank")),
        "original_blocker_delta_mean": original_blocker_delta,
        "base_top_blockers": base.get("class_top_blockers_compact"),
        "intervened_top_blockers": intervened.get("class_top_blockers_compact"),
        "base_top_tokens": base.get("top_tokens_compact"),
        "intervened_top_tokens": intervened.get("top_tokens_compact"),
    }


def group_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n": len(rows),
        "models": dict(Counter(str(row.get("model")) for row in rows)),
        "domains": dict(Counter(str(row.get("domain")) for row in rows)),
        "candidate_counts": dict(Counter(str(row.get("candidate_key")) for row in rows)),
        "closure_from_open": sum(1 for row in rows if row.get("closure_from_open")),
        "answer_gain": sum(1 for row in rows if row.get("answer_gain")),
        "clean_like_closure": sum(1 for row in rows if row.get("clean_like_closure")),
        "nonclean_like_closure": sum(1 for row in rows if row.get("nonclean_like_closure")),
        "intervened_boundary_closed": sum(1 for row in rows if row.get("intervened_class_boundary_closed")),
        "mean_class_logit_delta": mean([finite(row.get("class_logit_delta")) for row in rows if row.get("class_logit_delta") is not None]),
        "mean_blocker_reduction": mean([finite(row.get("blocker_reduction")) for row in rows if row.get("blocker_reduction") is not None]),
        "mean_rank_improvement": mean([finite(row.get("rank_improvement")) for row in rows if row.get("rank_improvement") is not None]),
        "mean_original_blocker_delta": mean(
            [finite(row.get("original_blocker_delta_mean")) for row in rows if row.get("original_blocker_delta_mean") is not None]
        ),
    }


def grouped(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[str(row.get(key))].append(row)
    return {name: group_summary(items) for name, items in sorted(buckets.items())}


def summarize_model(model_name: str, candidates: list[dict[str, Any]], rows: list[dict[str, Any]], status: str) -> dict[str, Any]:
    return {
        "phase": PHASE,
        "title": "Dominant Gear Robustness and Cross-Model Candidate Repair",
        "model": model_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "status": status,
        "candidate_count": len(candidates),
        "candidates": [
            {key: cand.get(key) for key in ("candidate_source", "candidate_key", "gear_key", "edit_mode", "source_hit_count")}
            for cand in candidates
        ],
        "n_rows": len(rows),
        "summary": group_summary(rows),
        "by_candidate": grouped(rows, "candidate_key"),
        "by_domain": grouped(rows, "domain"),
        "by_prompt": grouped(rows, "prompt_variant"),
        "boundary": (
            "Dominant single-gear robustness / candidate-repair audit. This is first-token boundary evidence, not language closure."
        ),
    }


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        f"# Phase 881 Dominant Gear Robustness and Repair ({payload.get('round')})",
        "",
        "- Boundary: single dominant gear robustness and candidate repair; not token-level minimal cut.",
        "- qwen3/GLM4 are included sequentially; missing candidate sources are recorded explicitly.",
        "",
        "## Models",
        "",
        "| model | status | candidates | rows | closure from open | answer gain | clean-like | nonclean-like |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for model_name in MODELS:
        summary = payload.get("model_summaries", {}).get(model_name) or {}
        s = summary.get("summary") or {}
        lines.append(
            f"| {model_name} | {summary.get('status', 'missing')} | {summary.get('candidate_count', 0)} | "
            f"{summary.get('n_rows', 0)} | {s.get('closure_from_open', 0)} | {s.get('answer_gain', 0)} | "
            f"{s.get('clean_like_closure', 0)} | {s.get('nonclean_like_closure', 0)} |"
        )
    lines += [
        "",
        "## Overall",
        "",
        f"- Overall summary: `{payload.get('overall_summary', {})}`",
        "",
        "## By Candidate",
        "",
        "| candidate | n | domains | closure from open | answer gain | nonclean-like | mean blocker red. | mean rank improve |",
        "|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for candidate, info in (payload.get("overall_by_candidate") or {}).items():
        lines.append(
            f"| `{candidate}` | {info.get('n', 0)} | `{info.get('domains', {})}` | "
            f"{info.get('closure_from_open', 0)} | {info.get('answer_gain', 0)} | {info.get('nonclean_like_closure', 0)} | "
            f"{info.get('mean_blocker_reduction')} | {info.get('mean_rank_improvement')} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def eval_model(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = args.output_root / args.round_name
    out_dir.mkdir(parents=True, exist_ok=True)
    modes = parse_csv(args.edit_modes)
    candidates = selected_candidates(args.model, modes, int(args.min_dominant_hits))
    cases = selected_cases_for_model(args.model, int(args.max_cases_per_domain))
    prompt_variants = parse_csv(args.prompt_variants)
    if args.dry_run or not candidates:
        status = "no_candidate_sources" if not candidates else "dry_run"
        payload = summarize_model(args.model, candidates, [], status)
        payload["case_count"] = len(cases)
        payload["prompt_variants"] = prompt_variants
        p846.write_json(out_dir / f"phase881_{args.model}_summary.json", payload)
        p846.write_jsonl(out_dir / f"phase881_{args.model}_rows.jsonl", [])
        print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
        return payload

    model = None
    tokenizer = None
    attn_impl = None
    rows: list[dict[str, Any]] = []
    try:
        model, tokenizer, device, attn_impl = p862.p844.p828.p796.load_model_bf16_prefer_flash(
            args.model, args.attn_implementations
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        base_cache: dict[tuple[str, str], tuple[dict[str, Any], torch.Tensor, list[dict[str, Any]]]] = {}
        for case_idx, case in enumerate(cases, 1):
            for prompt_variant in prompt_variants:
                base_key = (str(case["case_id"]), str(prompt_variant))
                if base_key not in base_cache:
                    prompt = p876.validation_prompt(case, prompt_variant)
                    prompt_ids = p862.p844.encode_prompt(tokenizer, prompt)
                    sets = p856.token_sets(tokenizer, case)
                    original_logits = p862.first_logits_with_scaled_gears(
                        model, device, prompt_ids, [], "original", float(args.scale_up_factor)
                    )
                    original_blockers = p862.p854.blocker_metrics(
                        tokenizer, original_logits, sets, int(args.topk_blockers)
                    ).get("class_top_blockers") or []
                    base = eval_condition(
                        model,
                        tokenizer,
                        device,
                        case,
                        prompt_variant,
                        None,
                        "original",
                        float(args.scale_up_factor),
                        int(args.max_new_tokens),
                        int(args.topk_tokens),
                        int(args.topk_blockers),
                        None,
                        None,
                    )
                    base_cache[base_key] = (base, original_logits, original_blockers)
                base, original_logits, original_blockers = base_cache[base_key]
                for cand in candidates:
                    row = eval_condition(
                        model,
                        tokenizer,
                        device,
                        case,
                        prompt_variant,
                        cand["gear"],
                        str(cand["edit_mode"]),
                        float(args.scale_up_factor),
                        int(args.max_new_tokens),
                        int(args.topk_tokens),
                        int(args.topk_blockers),
                        original_logits,
                        original_blockers,
                    )
                    rows.append(make_result_row(args.model, cand, case, prompt_variant, base, row))
            log(f"{args.model}/{args.round_name}: case={case_idx}/{len(cases)} rows={len(rows)}")
    finally:
        if model is not None:
            p862.p844.p828.release_model(model)
        if tokenizer is not None:
            del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    payload = summarize_model(args.model, candidates, rows, "complete")
    payload["case_count"] = len(cases)
    payload["prompt_variants"] = prompt_variants
    payload["attn_implementation"] = attn_impl
    payload["dtype"] = "bfloat16"
    payload["quantization"] = "off"
    p846.write_json(out_dir / f"phase881_{args.model}_summary.json", payload)
    p846.write_jsonl(out_dir / f"phase881_{args.model}_rows.jsonl", rows)
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
    return payload


def summarize_round(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = args.output_root / args.round_name
    payload: dict[str, Any] = {
        "phase": PHASE,
        "title": "Dominant Gear Robustness and Cross-Model Candidate Repair",
        "round": args.round_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "status": "missing",
        "models": [],
        "model_summaries": {},
        "all_rows": [],
    }
    for model_name in MODELS:
        path = out_dir / f"phase881_{model_name}_summary.json"
        rows_path = out_dir / f"phase881_{model_name}_rows.jsonl"
        if path.exists():
            summary = json.loads(path.read_text(encoding="utf-8"))
            payload["models"].append(model_name)
            payload["model_summaries"][model_name] = summary
        if rows_path.exists():
            payload["all_rows"].extend(read_jsonl(rows_path))
    payload["status"] = "complete" if len(payload["models"]) == len(MODELS) else "partial"
    payload["overall_summary"] = group_summary(payload["all_rows"])
    payload["overall_by_candidate"] = grouped(payload["all_rows"], "candidate_key")
    payload["overall_by_model"] = grouped(payload["all_rows"], "model")
    payload["boundary"] = (
        "This phase tests dominant single-gear robustness and candidate repair. It does not prove cross-model invariance unless "
        "candidate sources exist and close boundaries across models."
    )
    p846.write_json(out_dir / "phase881_cross_model_summary.json", payload)
    write_markdown(out_dir / "phase881_cross_model_summary.md", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Phase 881 dominant gear robustness and cross-model repair.")
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--output-root", type=Path, default=RESULT_ROOT)
    parser.add_argument("--round-name", default="dominant_l27c16651_repair")
    parser.add_argument("--edit-modes", default="flip,half,zero,scale_up")
    parser.add_argument("--prompt-variants", default="nonclean_direct,semantic_pressure,echo_pressure,format_pressure")
    parser.add_argument("--max-cases-per-domain", type=int, default=6)
    parser.add_argument("--min-dominant-hits", type=int, default=3)
    parser.add_argument("--scale-up-factor", type=float, default=2.0)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--topk-tokens", type=int, default=20)
    parser.add_argument("--topk-blockers", type=int, default=20)
    parser.add_argument("--attn-implementations", default="flash_attention_2,sdpa")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summarize-round", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.summarize_round:
        summarize_round(args)
        return
    if not args.model:
        raise SystemExit("--model is required unless --summarize-round is set")
    eval_model(args)


if __name__ == "__main__":
    main()
