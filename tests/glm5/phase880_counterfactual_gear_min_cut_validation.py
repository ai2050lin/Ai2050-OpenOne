#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import itertools
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


PHASE = 880
MODELS = p846.MODELS
RESULT_ROOT = Path("tests/result/phase880_counterfactual_gear_min_cut_validation")
DEFAULT_PHASE879_ROWS = Path(
    "tests/result/phase879_blocker_min_cut_proxy_audit/source_gate_phase876/phase879_min_cut_proxy_rows.jsonl"
)


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


def parse_candidate_key(candidate_key: str) -> tuple[list[dict[str, Any]], str]:
    text = str(candidate_key or "")
    if ":" in text:
        gear_part, mode = text.rsplit(":", 1)
    else:
        gear_part, mode = text, ""
    gears = []
    for key in gear_part.split("+"):
        if not key or key == "original":
            continue
        gear = p862.parse_gear_key(key)
        if gear is not None:
            gears.append(gear)
    return gears, mode


def gear_key(gear: dict[str, Any]) -> str:
    return p862.gear_key(gear)


def phase879_candidates(args: argparse.Namespace) -> list[dict[str, Any]]:
    classes = set(parse_csv(args.transition_classes))
    routes = set(parse_csv(args.routes))
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str]] = set()
    for row in read_jsonl(args.phase879_rows):
        if str(row.get("model")) != str(args.model):
            continue
        if classes and str(row.get("transition_class")) not in classes:
            continue
        if routes and str(row.get("primary_route")) not in routes:
            continue
        gears, mode = parse_candidate_key(str(row.get("candidate_key")))
        if len(gears) < 2:
            continue
        key = (str(row.get("case_id")), str(row.get("prompt_variant")), str(row.get("candidate_key")), str(row.get("primary_route")))
        if key in seen:
            continue
        seen.add(key)
        copied = dict(row)
        copied["gears"] = gears
        copied["mode"] = str(row.get("edit_mode") or mode)
        copied["gear_keys"] = [gear_key(gear) for gear in gears]
        rows.append(copied)
    if int(args.max_candidates) > 0:
        rows = rows[: int(args.max_candidates)]
    return rows


def validation_case_by_id() -> dict[str, dict[str, Any]]:
    return {str(case["case_id"]): case for case in p876.validation_cases()}


def top_token_compact(tokens: list[dict[str, Any]], limit: int = 8) -> list[dict[str, Any]]:
    out = []
    for item in (tokens or [])[: int(limit)]:
        out.append(
            {
                "token_id": item.get("token_id"),
                "token": item.get("token"),
                "role": item.get("role"),
                "logit": item.get("logit"),
                "gap_vs_threshold": item.get("gap_vs_threshold"),
            }
        )
    return out


def condition_boundary(row: dict[str, Any]) -> dict[str, Any]:
    class_count = row.get("blocker_class_blocker_count")
    class_rank = row.get("blocker_class_best_target_rank")
    strict_count = row.get("blocker_strict_blocker_count")
    strict_rank = row.get("blocker_strict_best_target_rank")
    return {
        "class_boundary_closed": class_count == 0 and class_rank == 1,
        "strict_boundary_closed": strict_count == 0 and strict_rank == 1,
        "class_target_rank": class_rank,
        "strict_target_rank": strict_rank,
        "class_blocker_count": class_count,
        "strict_blocker_count": strict_count,
        "class_target_logit": row.get("blocker_class_best_target_logit"),
        "strict_target_logit": row.get("blocker_strict_best_target_logit"),
    }


def eval_condition(
    model,
    tokenizer,
    device: torch.device,
    case: dict[str, Any],
    prompt_variant: str,
    gears: list[dict[str, Any]],
    mode: str,
    subset_name: str,
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
    effective_mode = "original" if not gears else mode
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
    blocker_deltas = {}
    if original_logits is not None and original_blockers is not None:
        blocker_deltas = p862.original_blocker_deltas(logits, original_logits, original_blockers, topk_blockers)
    row = {
        "condition_type": subset_name,
        "subset_name": subset_name,
        "edit_mode": effective_mode,
        "gear_count": len(gears),
        "gear_keys": [gear_key(gear) for gear in gears],
        "prompt": prompt,
        "token_ids": token_ids,
        "generated_clean": p856.clean_text(generated),
        **first,
        **{f"blocker_{key}": value for key, value in blocker.items()},
        **rollout,
        **blocker_deltas,
    }
    row.update(condition_boundary(row))
    row["class_top_blockers_compact"] = top_token_compact(row.get("blocker_class_top_blockers") or [])
    row["top_tokens_compact"] = top_token_compact(row.get("blocker_top_tokens") or row.get("top_tokens") or [])
    return row


def proper_subsets(gears: list[dict[str, Any]]) -> list[tuple[str, list[dict[str, Any]]]]:
    out: list[tuple[str, list[dict[str, Any]]]] = []
    for size in range(1, len(gears)):
        for idxs in itertools.combinations(range(len(gears)), size):
            subset = [gears[idx] for idx in idxs]
            name = "subset_" + "+".join(gear_key(gear) for gear in subset)
            out.append((name, subset))
    return out


def transition_changed(base: dict[str, Any], row: dict[str, Any]) -> bool:
    return str(base.get("rollout_label")) != str(row.get("rollout_label"))


def answer_like(row: dict[str, Any]) -> bool:
    return bool(row.get("rollout_answer_class") or row.get("rollout_clear_answer_class") or row.get("rollout_strict_canonical"))


def build_candidate_result(
    source: dict[str, Any],
    base: dict[str, Any],
    full: dict[str, Any],
    subset_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    closed_subsets = [row for row in subset_rows if row.get("class_boundary_closed")]
    answer_subsets = [row for row in subset_rows if answer_like(row)]
    full_closed = bool(full.get("class_boundary_closed"))
    full_answer = answer_like(full)
    base_closed = bool(base.get("class_boundary_closed"))
    gear_min_cut = bool(full_closed and not base_closed and not closed_subsets)
    answer_min_cut = bool(full_answer and not answer_like(base) and not answer_subsets)
    if not full_closed:
        minimality_class = "full_set_not_boundary_closed"
    elif closed_subsets:
        minimality_class = "proper_subset_also_boundary_closed"
    else:
        minimality_class = "gear_set_boundary_minimal_candidate"
    return {
        "phase": PHASE,
        "row_kind": "phase880_counterfactual_gear_min_cut_candidate",
        "model": source.get("model"),
        "domain": source.get("domain"),
        "case_id": source.get("case_id"),
        "object": source.get("object"),
        "prompt_variant": source.get("prompt_variant"),
        "candidate_key": source.get("candidate_key"),
        "gear_keys": source.get("gear_keys"),
        "edit_mode": source.get("mode"),
        "transition_class": source.get("transition_class"),
        "primary_route": source.get("primary_route"),
        "phase879_displacement_subtype": source.get("displacement_subtype"),
        "phase879_observed_proxy_closed": source.get("observed_proxy_closed"),
        "base_rollout_label": base.get("rollout_label"),
        "full_rollout_label": full.get("rollout_label"),
        "base_generated_clean": base.get("generated_clean"),
        "full_generated_clean": full.get("generated_clean"),
        "base_class_boundary_closed": base.get("class_boundary_closed"),
        "full_class_boundary_closed": full_closed,
        "full_answer_like": full_answer,
        "full_output_transition": transition_changed(base, full),
        "base_class_blocker_count": base.get("class_blocker_count"),
        "full_class_blocker_count": full.get("class_blocker_count"),
        "base_class_target_rank": base.get("class_target_rank"),
        "full_class_target_rank": full.get("class_target_rank"),
        "full_original_blocker_delta_mean": full.get("original_blocker_delta_mean"),
        "proper_subset_count": len(subset_rows),
        "proper_subset_boundary_closed_count": len(closed_subsets),
        "proper_subset_answer_like_count": len(answer_subsets),
        "proper_subset_boundary_closed_keys": [row.get("subset_name") for row in closed_subsets],
        "proper_subset_answer_like_keys": [row.get("subset_name") for row in answer_subsets],
        "gear_set_boundary_minimal_candidate": gear_min_cut,
        "gear_set_answer_minimal_candidate": answer_min_cut,
        "minimality_class": minimality_class,
        "conditions": {
            "base": compact_condition(base),
            "full_set": compact_condition(full),
            "proper_subsets": [compact_condition(row) for row in subset_rows],
        },
    }


def compact_condition(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "subset_name": row.get("subset_name"),
        "edit_mode": row.get("edit_mode"),
        "gear_keys": row.get("gear_keys"),
        "generated_clean": row.get("generated_clean"),
        "rollout_label": row.get("rollout_label"),
        "rollout_answer_class": row.get("rollout_answer_class"),
        "rollout_clear_answer_class": row.get("rollout_clear_answer_class"),
        "rollout_strict_canonical": row.get("rollout_strict_canonical"),
        "class_boundary_closed": row.get("class_boundary_closed"),
        "class_blocker_count": row.get("class_blocker_count"),
        "class_target_rank": row.get("class_target_rank"),
        "class_target_logit": row.get("class_target_logit"),
        "original_blocker_delta_mean": row.get("original_blocker_delta_mean"),
        "class_top_blockers": row.get("class_top_blockers_compact"),
        "top_tokens": row.get("top_tokens_compact"),
    }


def grouped_summary(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get(key))].append(row)
    return {name: summarize_rows(items) for name, items in sorted(groups.items())}


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n": len(rows),
        "transition_class_counts": dict(Counter(str(row.get("transition_class")) for row in rows)),
        "route_counts": dict(Counter(str(row.get("primary_route")) for row in rows)),
        "displacement_subtype_counts": dict(Counter(str(row.get("phase879_displacement_subtype")) for row in rows)),
        "minimality_class_counts": dict(Counter(str(row.get("minimality_class")) for row in rows)),
        "full_boundary_closed": sum(1 for row in rows if row.get("full_class_boundary_closed")),
        "full_answer_like": sum(1 for row in rows if row.get("full_answer_like")),
        "full_output_transition": sum(1 for row in rows if row.get("full_output_transition")),
        "gear_set_boundary_minimal_candidate": sum(1 for row in rows if row.get("gear_set_boundary_minimal_candidate")),
        "gear_set_answer_minimal_candidate": sum(1 for row in rows if row.get("gear_set_answer_minimal_candidate")),
        "proper_subset_boundary_closed_total": sum(int(row.get("proper_subset_boundary_closed_count") or 0) for row in rows),
        "proper_subset_answer_like_total": sum(int(row.get("proper_subset_answer_like_count") or 0) for row in rows),
        "mean_full_original_blocker_delta": mean(
            [finite(row.get("full_original_blocker_delta_mean")) for row in rows if row.get("full_original_blocker_delta_mean") is not None]
        ),
    }


def summarize_model(model_name: str, candidates: list[dict[str, Any]], rows: list[dict[str, Any]], status: str) -> dict[str, Any]:
    return {
        "phase": PHASE,
        "title": "Counterfactual Gear-Set Minimal Cut Validation",
        "model": model_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "status": status,
        "candidate_count": len(candidates),
        "n_rows": len(rows),
        "summary": summarize_rows(rows),
        "by_route": grouped_summary(rows, "primary_route"),
        "by_displacement_subtype": grouped_summary(rows, "phase879_displacement_subtype"),
        "boundary": (
            "Counterfactual validation over gear-set subsets, not token-level blocker minimal cut. "
            "Full logits are recomputed for tested gear subsets."
        ),
    }


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        f"# Phase 880 Counterfactual Gear-Set Minimal Cut Validation ({payload.get('round')})",
        "",
        "- Boundary: gear-subset counterfactual validation; not token-level blocker minimal cut.",
        "- Full next-token logits are recomputed for tested conditions.",
        "",
        "## Models",
        "",
        "| model | status | candidates | rows | full closed | gear-min candidates | subset closed |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for model_name in MODELS:
        summary = payload.get("model_summaries", {}).get(model_name) or {}
        s = summary.get("summary") or {}
        lines.append(
            f"| {model_name} | {summary.get('status', 'missing')} | {summary.get('candidate_count', 0)} | "
            f"{summary.get('n_rows', 0)} | {s.get('full_boundary_closed', 0)} | "
            f"{s.get('gear_set_boundary_minimal_candidate', 0)} | {s.get('proper_subset_boundary_closed_total', 0)} |"
        )
    lines += [
        "",
        "## Overall",
        "",
        f"- Overall summary: `{payload.get('overall_summary', {})}`",
        "",
        "## DS7B Rows",
        "",
        "| object | prompt | candidate | route | subtype | full closed | subset closed | minimality | base -> full |",
        "|---|---|---|---|---|---:|---:|---|---|",
    ]
    for row in payload.get("all_rows") or []:
        if row.get("model") != "deepseek7b":
            continue
        lines.append(
            f"| {row.get('object')} | {row.get('prompt_variant')} | `{row.get('candidate_key')}` | "
            f"{row.get('primary_route')} | {row.get('phase879_displacement_subtype')} | "
            f"{int(bool(row.get('full_class_boundary_closed')))} | {row.get('proper_subset_boundary_closed_count')} | "
            f"{row.get('minimality_class')} | `{row.get('base_rollout_label')} -> {row.get('full_rollout_label')}` |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def eval_model(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = args.output_root / args.round_name
    out_dir.mkdir(parents=True, exist_ok=True)
    candidates = phase879_candidates(args)
    rows: list[dict[str, Any]] = []
    if args.dry_run or not candidates:
        status = "no_phase879_candidates" if not candidates else "dry_run"
        payload = summarize_model(args.model, candidates, rows, status)
        p846.write_json(out_dir / f"phase880_{args.model}_summary.json", payload)
        p846.write_jsonl(out_dir / f"phase880_{args.model}_rows.jsonl", rows)
        print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
        return payload

    case_map = validation_case_by_id()
    model = None
    tokenizer = None
    attn_impl = None
    try:
        model, tokenizer, device, attn_impl = p862.p844.p828.p796.load_model_bf16_prefer_flash(
            args.model, args.attn_implementations
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        condition_cache: dict[tuple[str, str, str, tuple[str, ...]], dict[str, Any]] = {}

        def get_condition(case: dict[str, Any], prompt_variant: str, subset_name: str, gears: list[dict[str, Any]], mode: str) -> dict[str, Any]:
            key = (
                str(case["case_id"]),
                str(prompt_variant),
                str(mode if gears else "original"),
                tuple(gear_key(gear) for gear in gears),
            )
            if key in condition_cache:
                return condition_cache[key]
            base_key = (str(case["case_id"]), str(prompt_variant), "original", tuple())
            if gears and base_key not in condition_cache:
                condition_cache[base_key] = eval_condition(
                    model,
                    tokenizer,
                    device,
                    case,
                    prompt_variant,
                    [],
                    "original",
                    "base",
                    float(args.scale_up_factor),
                    int(args.max_new_tokens),
                    int(args.topk_tokens),
                    int(args.topk_blockers),
                    None,
                    None,
                )
            base = condition_cache.get(base_key)
            original_logits = None
            original_blockers = None
            if gears:
                # Recompute base logits once for blocker-delta reference. This keeps memory low and avoids storing tensors in cache.
                prompt = p876.validation_prompt(case, prompt_variant)
                prompt_ids = p862.p844.encode_prompt(tokenizer, prompt)
                original_logits = p862.first_logits_with_scaled_gears(
                    model, device, prompt_ids, [], "original", float(args.scale_up_factor)
                )
                sets = p856.token_sets(tokenizer, case)
                original_blockers = p862.p854.blocker_metrics(tokenizer, original_logits, sets, int(args.topk_blockers)).get(
                    "class_top_blockers"
                ) or []
            row = eval_condition(
                model,
                tokenizer,
                device,
                case,
                prompt_variant,
                gears,
                mode,
                subset_name,
                float(args.scale_up_factor),
                int(args.max_new_tokens),
                int(args.topk_tokens),
                int(args.topk_blockers),
                original_logits,
                original_blockers,
            )
            if base is not None and not gears:
                row = base
            condition_cache[key] = row
            return row

        for idx, source in enumerate(candidates, 1):
            case = case_map.get(str(source.get("case_id")))
            if case is None:
                log(f"{args.model}: skip missing case_id={source.get('case_id')}")
                continue
            prompt_variant = str(source.get("prompt_variant"))
            gears = list(source["gears"])
            mode = str(source["mode"])
            base = get_condition(case, prompt_variant, "base", [], "original")
            full = get_condition(case, prompt_variant, "full_set", gears, mode)
            subset_rows = [
                get_condition(case, prompt_variant, subset_name, subset_gears, mode)
                for subset_name, subset_gears in proper_subsets(gears)
            ]
            rows.append(build_candidate_result(source, base, full, subset_rows))
            log(f"{args.model}/{args.round_name}: {idx}/{len(candidates)} {source.get('candidate_key')} rows={len(rows)}")
    finally:
        if model is not None:
            p862.p844.p828.release_model(model)
        if tokenizer is not None:
            del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    payload = summarize_model(args.model, candidates, rows, "complete")
    payload["attn_implementation"] = attn_impl
    payload["dtype"] = "bfloat16"
    payload["quantization"] = "off"
    p846.write_json(out_dir / f"phase880_{args.model}_summary.json", payload)
    p846.write_jsonl(out_dir / f"phase880_{args.model}_rows.jsonl", rows)
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
    return payload


def summarize_round(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = args.output_root / args.round_name
    payload: dict[str, Any] = {
        "phase": PHASE,
        "title": "Counterfactual Gear-Set Minimal Cut Validation",
        "round": args.round_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "status": "missing",
        "models": [],
        "model_summaries": {},
        "all_rows": [],
    }
    for model_name in MODELS:
        path = out_dir / f"phase880_{model_name}_summary.json"
        rows_path = out_dir / f"phase880_{model_name}_rows.jsonl"
        if path.exists():
            summary = json.loads(path.read_text(encoding="utf-8"))
            payload["models"].append(model_name)
            payload["model_summaries"][model_name] = summary
        if rows_path.exists():
            payload["all_rows"].extend(read_jsonl(rows_path))
    payload["status"] = "complete" if len(payload["models"]) == len(MODELS) else "partial"
    payload["overall_summary"] = summarize_rows(payload["all_rows"])
    payload["boundary"] = (
        "This phase tests minimality over source gear subsets. It does not directly ablate blocker tokens and does not prove "
        "full language closure."
    )
    p846.write_json(out_dir / "phase880_cross_model_summary.json", payload)
    write_markdown(out_dir / "phase880_cross_model_summary.md", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Phase 880 counterfactual gear-set minimal cut validation.")
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--phase879-rows", type=Path, default=DEFAULT_PHASE879_ROWS)
    parser.add_argument("--output-root", type=Path, default=RESULT_ROOT)
    parser.add_argument("--round-name", default="gear_subset_phase879")
    parser.add_argument("--transition-classes", default="clean_causal_transition,nonclean_output_transition")
    parser.add_argument("--routes", default="")
    parser.add_argument("--max-candidates", type=int, default=0)
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
