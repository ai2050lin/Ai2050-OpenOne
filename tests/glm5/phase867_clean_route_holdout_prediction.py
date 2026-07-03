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
from model_utils import get_layers  # noqa: E402


PHASE = 867
MODELS = p846.MODELS
RESULT_ROOT = Path("tests/result/phase867_clean_route_holdout_prediction")
PHASE865_ROWS = Path("tests/result/phase865_route_purity_and_side_effect_filter/phase865_route_purity_rows.jsonl")


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def finite(value: Any, default: float = 0.0) -> float:
    return p846.finite(value, default)


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def parse_csv(text: str) -> list[str]:
    return [part.strip() for part in str(text or "").split(",") if part.strip()]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return p846.read_jsonl(path)


def gear_key(gear: dict[str, Any]) -> str:
    return p862.gear_key(gear)


def holdout_cases() -> list[dict[str, Any]]:
    raw = [
        ("animal", "horse", ["animal", "living thing", "living creature", "creature", "mammal"]),
        ("animal", "cow", ["animal", "living thing", "living creature", "creature", "mammal"]),
        ("animal", "lion", ["animal", "living thing", "living creature", "creature", "mammal"]),
        ("animal", "whale", ["animal", "living thing", "living creature", "creature", "mammal"]),
        ("material", "stone", ["material", "substance", "matter"]),
        ("material", "paper", ["material", "substance", "matter"]),
        ("material", "cotton", ["material", "substance", "matter"]),
        ("material", "rubber", ["material", "substance", "matter"]),
        ("color", "purple", ["color", "hue", "colour"]),
        ("color", "orange", ["color", "hue", "colour"]),
        ("color", "black", ["color", "hue", "colour"]),
        ("color", "white", ["color", "hue", "colour"]),
    ]
    cases: list[dict[str, Any]] = []
    for idx, (domain, obj, aliases) in enumerate(raw, 1):
        cases.append(
            {
                "case_id": f"p867_{idx:03d}_{domain}_{obj}",
                "domain": domain,
                "object": obj,
                "answer_aliases": aliases,
                "canonical_answer": aliases[0],
                "overlap_kind": "holdout_member_not_alias",
            }
        )
    return cases


def prompt_for_holdout(case: dict[str, Any], variant: str) -> str:
    obj = case["object"]
    if variant == "holdout_category_short":
        return f"Give only the category phrase.\nItem = {obj}\nCategory:"
    if variant == "holdout_kind_phrase":
        return f"Complete with a short category phrase.\n{obj} is a kind of"
    if variant == "holdout_label":
        return f"Object: {obj}\nBest category phrase:"
    raise ValueError(f"unknown holdout prompt variant: {variant}")


def selected_candidates(model_name: str, include_non_clean_controls: bool) -> list[dict[str, Any]]:
    if not PHASE865_ROWS.exists():
        raise FileNotFoundError(f"missing Phase 865 rows: {PHASE865_ROWS}")
    rows = []
    for row in read_jsonl(PHASE865_ROWS):
        if str(row.get("model")) != model_name:
            continue
        if str(row.get("condition_type")) != "full_set" or str(row.get("subset_name")) != "full":
            continue
        if not include_non_clean_controls and str(row.get("purity_class")) != "clean_mixed_answer_blocker_route":
            continue
        gears = [p862.parse_gear_key(str(key)) for key in row.get("gear_keys") or []]
        gears = [gear for gear in gears if gear is not None]
        if not gears:
            continue
        copied = dict(row)
        copied["gears"] = gears
        copied["source_predict_clean_mixed"] = str(row.get("purity_class")) == "clean_mixed_answer_blocker_route"
        copied["candidate_key"] = "+".join(gear_key(gear) for gear in gears) + f":{row.get('edit_mode')}"
        rows.append(copied)
    rows.sort(key=lambda row: (str(row.get("domain")), str(row.get("edit_mode"))))
    return rows


def selected_cases(domains: set[str], max_cases_per_domain: int) -> list[dict[str, Any]]:
    counts: Counter[str] = Counter()
    out: list[dict[str, Any]] = []
    for case in holdout_cases():
        domain = str(case["domain"])
        if domains and domain not in domains:
            continue
        if int(max_cases_per_domain) > 0 and counts[domain] >= int(max_cases_per_domain):
            continue
        out.append(case)
        counts[domain] += 1
    return out


def row_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (str(row.get("domain")), str(row.get("case_id")), str(row.get("prompt_variant")))


def pair_effects(rows: list[dict[str, Any]], object_delta_threshold: float) -> list[dict[str, Any]]:
    originals = {row_key(row): row for row in rows if row.get("condition_type") == "original"}
    grouped: dict[tuple[str, str, str, str], list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for row in rows:
        if row.get("condition_type") == "original":
            continue
        base = originals.get(row_key(row))
        if base is None:
            continue
        grouped[
            (
                str(row.get("model")),
                str(row.get("domain")),
                str(row.get("candidate_key")),
                str(row.get("edit_mode")),
            )
        ].append((base, row))

    out: list[dict[str, Any]] = []
    for (model_name, domain, candidate_key, edit_mode), pairs in sorted(grouped.items()):
        first = pairs[0][1]
        clear_gain = sum(1 for base, row in pairs if not base.get("rollout_clear_answer_class") and row.get("rollout_clear_answer_class"))
        clear_loss = sum(1 for base, row in pairs if base.get("rollout_clear_answer_class") and not row.get("rollout_clear_answer_class"))
        rollout_gain = sum(1 for base, row in pairs if not base.get("rollout_answer_class") and row.get("rollout_answer_class"))
        rollout_loss = sum(1 for base, row in pairs if base.get("rollout_answer_class") and not row.get("rollout_answer_class"))
        object_echo_induced = sum(1 for base, row in pairs if not base.get("rollout_object_echo") and row.get("rollout_object_echo"))
        object_echo_reduced = sum(1 for base, row in pairs if base.get("rollout_object_echo") and not row.get("rollout_object_echo"))
        format_or_other_induced = sum(
            1 for base, row in pairs if not base.get("rollout_other_or_format") and row.get("rollout_other_or_format")
        )
        format_or_other_reduced = sum(
            1 for base, row in pairs if base.get("rollout_other_or_format") and not row.get("rollout_other_or_format")
        )
        blocker_reduction = [
            finite(base.get("class_blocker_count")) - finite(row.get("class_blocker_count"))
            for base, row in pairs
            if base.get("class_blocker_count") is not None and row.get("class_blocker_count") is not None
        ]
        clear_blocker_reduction = [
            finite(base.get("clear_class_blocker_count")) - finite(row.get("clear_class_blocker_count"))
            for base, row in pairs
            if base.get("clear_class_blocker_count") is not None and row.get("clear_class_blocker_count") is not None
        ]
        answer_delta = [finite(row.get("class_answer_delta")) for _, row in pairs if row.get("class_answer_delta") is not None]
        object_delta = [finite(row.get("object_delta")) for _, row in pairs if row.get("object_delta") is not None]
        original_blocker_delta = [
            finite(row.get("original_blocker_delta_mean")) for _, row in pairs if row.get("original_blocker_delta_mean") is not None
        ]
        mean_answer = mean(answer_delta)
        mean_blocker = mean(blocker_reduction)
        mean_orig_blocker = mean(original_blocker_delta)
        mean_object = mean(object_delta)
        formula_clean = bool(
            (mean_answer or 0.0) > 0.0
            and (mean_blocker or 0.0) > 0.0
            and (mean_orig_blocker or 0.0) < 0.0
            and (mean_object if mean_object is not None else 999.0) <= float(object_delta_threshold)
            and object_echo_induced == 0
            and format_or_other_induced == 0
        )
        holdout_clean = bool(formula_clean and clear_gain > 0 and clear_loss == 0)
        out.append(
            {
                "model": model_name,
                "domain": domain,
                "candidate_key": candidate_key,
                "edit_mode": edit_mode,
                "gear_keys": first.get("gear_keys"),
                "source_purity_class": first.get("source_purity_class"),
                "source_predict_clean_mixed": bool(first.get("source_predict_clean_mixed")),
                "n_pairs": len(pairs),
                "n_cases": len({row.get("case_id") for _, row in pairs}),
                "n_prompts": len({row.get("prompt_variant") for _, row in pairs}),
                "clear_rollout_gain": clear_gain,
                "clear_rollout_loss": clear_loss,
                "rollout_gain": rollout_gain,
                "rollout_loss": rollout_loss,
                "object_echo_induced": object_echo_induced,
                "object_echo_reduced": object_echo_reduced,
                "format_or_other_induced": format_or_other_induced,
                "format_or_other_reduced": format_or_other_reduced,
                "mean_answer_delta": mean_answer,
                "mean_class_blocker_reduction": mean_blocker,
                "mean_clear_blocker_reduction": mean(clear_blocker_reduction),
                "mean_original_blocker_delta": mean_orig_blocker,
                "mean_object_delta": mean_object,
                "formula_clean_mixed": formula_clean,
                "holdout_clean_mixed": holdout_clean,
            }
        )
    return out


def binary_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    tp = sum(1 for row in rows if row.get("source_predict_clean_mixed") and row.get("holdout_clean_mixed"))
    fp = sum(1 for row in rows if row.get("source_predict_clean_mixed") and not row.get("holdout_clean_mixed"))
    fn = sum(1 for row in rows if not row.get("source_predict_clean_mixed") and row.get("holdout_clean_mixed"))
    tn = sum(1 for row in rows if not row.get("source_predict_clean_mixed") and not row.get("holdout_clean_mixed"))
    n = len(rows)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "n": n,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "accuracy": (tp + tn) / n if n else 0.0,
        "source_clean_count": sum(1 for row in rows if row.get("source_predict_clean_mixed")),
        "holdout_clean_count": sum(1 for row in rows if row.get("holdout_clean_mixed")),
    }


def eval_model(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = RESULT_ROOT / args.round_name
    out_dir.mkdir(parents=True, exist_ok=True)
    candidates = selected_candidates(args.model, bool(args.include_non_clean_controls))
    domains = {str(row.get("domain")) for row in candidates}
    cases = selected_cases(domains, int(args.max_cases_per_domain))
    prompt_variants = parse_csv(args.prompt_variants)
    if args.dry_run or not candidates:
        payload = {
            "phase": PHASE,
            "title": "Clean Route Holdout Prediction",
            "model": args.model,
            "round": args.round_name,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "status": "no_phase865_candidates" if not candidates else "dry_run",
            "candidate_count": len(candidates),
            "holdout_case_count": len(cases),
            "prompt_variants": prompt_variants,
        }
        p846.write_json(out_dir / f"phase867_{args.model}_summary.json", payload)
        p846.write_jsonl(out_dir / f"phase867_{args.model}_rows.jsonl", [])
        print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
        return payload

    model = None
    tokenizer = None
    rows: list[dict[str, Any]] = []
    attn_impl = None
    try:
        model, tokenizer, device, attn_impl = p862.p844.p828.p796.load_model_bf16_prefer_flash(
            args.model, args.attn_implementations
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        n_layers = len(get_layers(model))
        for domain in sorted(domains):
            domain_cases = [case for case in cases if str(case.get("domain")) == domain]
            domain_candidates = [row for row in candidates if str(row.get("domain")) == domain]
            for case_idx, case in enumerate(domain_cases, 1):
                sets = p856.token_sets(tokenizer, case)
                for prompt_variant in prompt_variants:
                    prompt = prompt_for_holdout(case, prompt_variant)
                    prompt_ids = p862.p844.encode_prompt(tokenizer, prompt)
                    original_logits = p862.first_logits_with_scaled_gears(
                        model, device, prompt_ids, [], "original", float(args.scale_up_factor)
                    )
                    original_first = p856.first_token_metrics(tokenizer, original_logits, sets, int(args.topk_tokens))
                    original_blockers = p862.p854.blocker_metrics(tokenizer, original_logits, sets, int(args.topk_blockers))
                    original_generated, original_token_ids = p862.greedy_with_scaled_gears(
                        model,
                        tokenizer,
                        device,
                        prompt_ids,
                        [],
                        "original",
                        int(args.max_new_tokens),
                        float(args.scale_up_factor),
                    )
                    original_rollout = p856.classify_rollout(original_generated, case)
                    class_score_orig, class_id_orig = p862.best_score_for_ids(original_logits, sets["class_target_ids"])
                    object_score_orig, object_id_orig = p862.best_score_for_ids(original_logits, sets["object_ids"])
                    strict_score_orig, strict_id_orig = p862.best_score_for_ids(original_logits, sets["strict_target_ids"])
                    base_common = {
                        "row_kind": "phase867_clean_route_holdout_prediction",
                        "phase": PHASE,
                        "model": args.model,
                        "round": args.round_name,
                        "domain": domain,
                        "case_id": case["case_id"],
                        "object": case["object"],
                        "canonical_answer": case["canonical_answer"],
                        "prompt_variant": prompt_variant,
                        "prompt": prompt,
                        **sets,
                    }
                    rows.append(
                        {
                            **base_common,
                            "condition_type": "original",
                            "candidate_key": "original",
                            "edit_mode": "original",
                            "gear_count": 0,
                            "gear_keys": [],
                            "token_ids": original_token_ids,
                            "generated_clean": p856.clean_text(original_generated),
                            **original_first,
                            **{f"blocker_{k}": v for k, v in original_blockers.items()},
                            **original_rollout,
                            "class_answer_delta": 0.0,
                            "object_delta": 0.0,
                            "strict_delta": 0.0,
                            "original_blocker_delta_mean": 0.0,
                            "original_blocker_delta_top1": 0.0,
                            "original_blocker_delta_negative_count": 0,
                            "original_blocker_delta_positive_count": 0,
                        }
                    )
                    for cand in domain_candidates:
                        valid_gears = [
                            gear
                            for gear in cand["gears"]
                            if 0 <= int(gear["layer_idx"]) < n_layers and int(gear["channel_id"]) >= 0
                        ]
                        mode = str(cand.get("edit_mode"))
                        logits = p862.first_logits_with_scaled_gears(
                            model, device, prompt_ids, valid_gears, mode, float(args.scale_up_factor)
                        )
                        first = p856.first_token_metrics(tokenizer, logits, sets, int(args.topk_tokens))
                        blocker = p862.p854.blocker_metrics(tokenizer, logits, sets, int(args.topk_blockers))
                        generated, token_ids = p862.greedy_with_scaled_gears(
                            model,
                            tokenizer,
                            device,
                            prompt_ids,
                            valid_gears,
                            mode,
                            int(args.max_new_tokens),
                            float(args.scale_up_factor),
                        )
                        rollout = p856.classify_rollout(generated, case)
                        class_score, _ = p862.best_score_for_ids(logits, sets["class_target_ids"])
                        object_score, _ = p862.best_score_for_ids(logits, sets["object_ids"])
                        strict_score, _ = p862.best_score_for_ids(logits, sets["strict_target_ids"])
                        blocker_deltas = p862.original_blocker_deltas(
                            logits,
                            original_logits,
                            original_blockers.get("class_top_blockers") or [],
                            int(args.topk_blockers),
                        )
                        rows.append(
                            {
                                **base_common,
                                "condition_type": "full_set",
                                "candidate_key": cand["candidate_key"],
                                "edit_mode": mode,
                                "source_purity_class": cand.get("purity_class"),
                                "source_route_class": cand.get("route_class"),
                                "source_predict_clean_mixed": bool(cand.get("source_predict_clean_mixed")),
                                "source_clear_gain": cand.get("clear_gain"),
                                "source_clear_loss": cand.get("clear_loss"),
                                "scale_up_factor": float(args.scale_up_factor),
                                "gear_count": len(valid_gears),
                                "gear_keys": [gear_key(gear) for gear in valid_gears],
                                "token_ids": token_ids,
                                "generated_clean": p856.clean_text(generated),
                                **first,
                                **{f"blocker_{k}": v for k, v in blocker.items()},
                                **rollout,
                                "class_answer_delta": None
                                if class_score is None or class_score_orig is None
                                else float(class_score - class_score_orig),
                                "object_delta": None
                                if object_score is None or object_score_orig is None
                                else float(object_score - object_score_orig),
                                "strict_delta": None
                                if strict_score is None or strict_score_orig is None
                                else float(strict_score - strict_score_orig),
                                "original_class_token_delta": p862.token_delta(logits, original_logits, class_id_orig),
                                "original_object_token_delta": p862.token_delta(logits, original_logits, object_id_orig),
                                "original_strict_token_delta": p862.token_delta(logits, original_logits, strict_id_orig),
                                **blocker_deltas,
                            }
                        )
                log(f"{args.model}/{args.round_name}: domain={domain} case={case_idx}/{len(domain_cases)} rows={len(rows)}")
    finally:
        if model is not None:
            p862.p844.p828.release_model(model)
        if tokenizer is not None:
            del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    effects = pair_effects(rows, float(args.object_delta_threshold))
    stats = binary_stats(effects)
    summary = {
        "phase": PHASE,
        "title": "Clean Route Holdout Prediction",
        "model": args.model,
        "round": args.round_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "status": "complete",
        "attn_implementation": attn_impl,
        "dtype": "bfloat16",
        "quantization": "off",
        "object_delta_threshold": float(args.object_delta_threshold),
        "candidate_count": len(candidates),
        "holdout_case_count": len(cases),
        "prompt_variants": prompt_variants,
        "domains": sorted(domains),
        "n_rows": len(rows),
        "holdout_effects": effects,
        "source_to_holdout_clean_stats": stats,
        "boundary": (
            "Holdout validation of Phase 866 clean route rule over new objects/prompts. "
            "The rule and threshold are fixed; this is not language closure."
        ),
    }
    p846.write_jsonl(out_dir / f"phase867_{args.model}_rows.jsonl", rows)
    p846.write_jsonl(out_dir / f"phase867_{args.model}_effects.jsonl", effects)
    p846.write_json(out_dir / f"phase867_{args.model}_summary.json", summary)
    print(
        json.dumps(
            {
                "model": args.model,
                "round": args.round_name,
                "status": "complete",
                "rows": len(rows),
                "domains": sorted(domains),
                "source_to_holdout_clean_stats": stats,
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return summary


def summarize_round(round_name: str) -> dict[str, Any]:
    out_dir = RESULT_ROOT / round_name
    payload: dict[str, Any] = {
        "phase": PHASE,
        "round": round_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "status": "missing",
        "models": [],
        "model_summaries": {},
    }
    all_effects: list[dict[str, Any]] = []
    for model_name in MODELS:
        path = out_dir / f"phase867_{model_name}_summary.json"
        if path.exists():
            summary = json.loads(path.read_text(encoding="utf-8"))
            payload["models"].append(model_name)
            payload["model_summaries"][model_name] = summary
            all_effects.extend(summary.get("holdout_effects") or [])
    payload["status"] = "complete" if len(payload["models"]) == len(MODELS) else "partial"
    payload["overall_source_to_holdout_clean_stats"] = binary_stats(all_effects)
    p846.write_json(out_dir / "phase867_cross_model_summary.json", payload)
    write_markdown(out_dir / "phase867_cross_model_summary.md", payload)
    return payload


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        f"# Phase 867 Clean Route Holdout Prediction ({payload['round']})",
        "",
        "- Source: Phase 865 full-set route purity rows.",
        "- Fixed rule: Phase 866 CleanMixedRoute, object_delta_threshold=0.25 unless configured.",
        "- Boundary: holdout rule validation, not language closure.",
        "",
        "## Cross-Model Summary",
        "",
        "| model | status | candidates | domains | source-clean -> holdout-clean stats |",
        "|---|---|---:|---|---|",
    ]
    for model_name in MODELS:
        summary = payload.get("model_summaries", {}).get(model_name) or {}
        lines.append(
            f"| {model_name} | {summary.get('status', 'missing')} | {summary.get('candidate_count', 0)} | "
            f"`{summary.get('domains', [])}` | `{summary.get('source_to_holdout_clean_stats', {})}` |"
        )
    lines += [
        "",
        "## Holdout Effects",
        "",
        "| model | domain | mode | source purity | source clean | holdout clean | clear gain/loss | ans delta | blocker red. | orig blocker delta | object delta | side effects |",
        "|---|---|---|---|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for model_name in MODELS:
        summary = payload.get("model_summaries", {}).get(model_name) or {}
        for row in summary.get("holdout_effects") or []:
            lines.append(
                f"| {model_name} | {row.get('domain')} | `{row.get('edit_mode')}` | "
                f"`{row.get('source_purity_class')}` | {row.get('source_predict_clean_mixed')} | "
                f"{row.get('holdout_clean_mixed')} | "
                f"{row.get('clear_rollout_gain', 0)}/{row.get('clear_rollout_loss', 0)} | "
                f"{fmt(row.get('mean_answer_delta'))} | "
                f"{fmt(row.get('mean_class_blocker_reduction'))} | "
                f"{fmt(row.get('mean_original_blocker_delta'))} | "
                f"{fmt(row.get('mean_object_delta'))} | "
                f"echo+{row.get('object_echo_induced', 0)}, fmt+{row.get('format_or_other_induced', 0)} |"
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--round-name", default="main")
    parser.add_argument("--max-cases-per-domain", type=int, default=4)
    parser.add_argument("--prompt-variants", default="holdout_category_short,holdout_kind_phrase,holdout_label")
    parser.add_argument("--include-non-clean-controls", action="store_true")
    parser.add_argument("--object-delta-threshold", type=float, default=0.25)
    parser.add_argument("--scale-up-factor", type=float, default=2.0)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--topk-tokens", type=int, default=20)
    parser.add_argument("--topk-blockers", type=int, default=10)
    parser.add_argument("--attn-implementations", default="flash_attention_2,sdpa")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summarize-round", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.summarize_round:
        payload = summarize_round(args.round_name)
        print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
        return
    if not args.model:
        raise SystemExit("--model is required unless --summarize-round is set")
    eval_model(args)


if __name__ == "__main__":
    main()
