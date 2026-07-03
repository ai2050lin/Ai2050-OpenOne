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

import phase844_geometry_route_natural_gear_set_search as p844  # noqa: E402
import phase846_geometry_boundary_equation_fitting as p846  # noqa: E402
import phase854_full_vocab_blocker_min_cut_validation as p854  # noqa: E402
import phase856_identity_class_overlap_cross_domain_rollout_audit as p856  # noqa: E402
import phase858_cross_domain_independent_gear_isomorphism_audit as p858  # noqa: E402
from model_utils import get_layers  # noqa: E402


PHASE = 859
MODELS = p846.MODELS
RESULT_ROOT = Path("tests/result/phase859_domain_gear_holdout_sign_calibration_audit")
PHASE858_ROOT = Path("tests/result/phase858_cross_domain_independent_gear_isomorphism_audit")


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def finite(value: Any, default: float = 0.0) -> float:
    return p846.finite(value, default)


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def parse_csv(text: str) -> list[str]:
    return [part.strip() for part in str(text or "").split(",") if part.strip()]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return p846.read_jsonl(path)


def gear_from_key(key: str) -> dict[str, Any] | None:
    return p854.gear_from_key(key)


def gear_key(gear: dict[str, Any]) -> str:
    return p854.gear_key(gear)


def phase858_summary(model_name: str, round_name: str) -> dict[str, Any]:
    path = PHASE858_ROOT / round_name / f"phase858_{model_name}_summary.json"
    if not path.exists():
        raise FileNotFoundError(f"missing Phase 858 summary: {path}")
    return read_json(path)


def phase858_candidates(model_name: str, round_name: str) -> list[dict[str, Any]]:
    path = PHASE858_ROOT / round_name / f"phase858_{model_name}_candidate_gears.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"missing Phase 858 candidates: {path}")
    return read_jsonl(path)


def selected_holdout_cases(args: argparse.Namespace, used_cases: set[str]) -> list[dict[str, Any]]:
    domains = set(parse_csv(args.domains))
    rows = [dict(case) for case in p856.base_cases() if not domains or str(case.get("domain")) in domains]
    out: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for case in rows:
        domain = str(case.get("domain"))
        if str(case.get("case_id")) in used_cases:
            continue
        if counts[domain] >= int(args.max_holdout_cases_per_domain):
            continue
        out.append(case)
        counts[domain] += 1
    if int(args.max_cases) > 0:
        out = out[: int(args.max_cases)]
    return out


def used_phase858_cases(model_name: str, round_name: str) -> set[str]:
    path = PHASE858_ROOT / round_name / f"phase858_{model_name}_rows.jsonl"
    if not path.exists():
        return set()
    rows = read_jsonl(path)
    return {str(row.get("case_id")) for row in rows}


def shared_gear_specs(summary: dict[str, Any]) -> list[dict[str, Any]]:
    shared = ((summary.get("isomorphism_audit") or {}).get("shared_best_gears") or {})
    out: list[dict[str, Any]] = []
    effects = summary.get("candidate_effects") or []
    mode_by_gear: dict[str, str] = {}
    for row in effects:
        for key in row.get("gear_keys") or []:
            mode_by_gear.setdefault(str(key), str(row.get("mode") or "flip"))
    for key, domains in shared.items():
        gear = gear_from_key(str(key))
        if gear is None:
            continue
        out.append(
            {
                "condition_type": "shared_exact_probe",
                "candidate_key": f"{key}:shared_exact_probe",
                "candidate_role": "shared_exact_probe",
                "source_domains": domains,
                "mode": mode_by_gear.get(str(key), "flip"),
                "gears": [gear],
            }
        )
    return out


def same_layer_control(
    domain: str,
    best_gear_keys: list[str],
    best_mode: str,
    candidates: list[dict[str, Any]],
) -> dict[str, Any] | None:
    parsed = [p854.parse_gear_key(key) for key in best_gear_keys]
    layers = [layer for layer, _ in parsed if layer is not None] if parsed else []
    used = set(best_gear_keys)
    controls: list[dict[str, Any]] = []
    for layer in layers:
        rows = [
            row
            for row in candidates
            if str(row.get("domain")) == domain
            and int(row.get("layer_idx", -1)) == int(layer)
            and str(row.get("gear_key")) not in used
        ]
        rows.sort(key=lambda row: finite(row.get("abs_support_mean")), reverse=True)
        if rows:
            gear = gear_from_key(str(rows[0].get("gear_key")))
            if gear is not None:
                controls.append(gear)
    if not controls:
        return None
    return {
        "condition_type": "same_layer_control",
        "candidate_key": "+".join(gear_key(gear) for gear in controls) + f":same_layer_control:{best_mode}",
        "candidate_role": "same_layer_control",
        "mode": best_mode,
        "gears": controls,
    }


def condition_specs_for_domain(domain: str, summary: dict[str, Any], candidates: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    best = ((summary.get("isomorphism_audit") or {}).get("best_by_domain") or {}).get(domain)
    specs: list[dict[str, Any]] = [
        {"condition_type": "original", "candidate_key": "original", "candidate_role": "baseline", "mode": "original", "gears": []}
    ]
    if best:
        best_keys = [str(key) for key in best.get("gear_keys") or []]
        best_gears = [gear_from_key(key) for key in best_keys]
        best_gears = [gear for gear in best_gears if gear is not None]
        best_mode = str(best.get("mode") or "flip")
        if best_gears:
            specs.append(
                {
                    "condition_type": "best_holdout",
                    "candidate_key": str(best.get("candidate_key") or "+".join(best_keys)),
                    "candidate_role": str(best.get("candidate_role") or "best"),
                    "mode": best_mode,
                    "gears": best_gears,
                    "source_effect_score": best.get("effect_score"),
                    "source_clear_gain": best.get("clear_rollout_gain"),
                    "source_clear_loss": best.get("clear_rollout_loss"),
                }
            )
            alt_mode = "zero" if best_mode == "flip" else "flip"
            specs.append(
                {
                    "condition_type": "alternate_mode",
                    "candidate_key": "+".join(best_keys) + f":alternate:{alt_mode}",
                    "candidate_role": str(best.get("candidate_role") or "best"),
                    "mode": alt_mode,
                    "gears": best_gears,
                    "source_effect_score": best.get("effect_score"),
                }
            )
            control = same_layer_control(domain, best_keys, best_mode, candidates)
            if control:
                specs.append(control)
    if args.include_shared_exact_probe:
        for spec in shared_gear_specs(summary):
            specs.append(spec)
    return specs


def row_pair_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (str(row.get("domain")), str(row.get("case_id")), str(row.get("prompt_variant")))


def compact(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n": len(rows),
        "first_token_answer_class": sum(1 for row in rows if row.get("first_token_answer_class")),
        "first_token_clear_answer_class": sum(1 for row in rows if row.get("first_token_clear_answer_class")),
        "rollout_answer_class": sum(1 for row in rows if row.get("rollout_answer_class")),
        "rollout_clear_answer_class": sum(1 for row in rows if row.get("rollout_clear_answer_class")),
        "rollout_object_echo": sum(1 for row in rows if row.get("rollout_object_echo")),
        "rollout_other_or_format": sum(1 for row in rows if row.get("rollout_other_or_format")),
        "rollout_labels": dict(Counter(str(row.get("rollout_label")) for row in rows)),
        "mean_class_blocker_count": mean(
            [finite(row.get("class_blocker_count")) for row in rows if row.get("class_blocker_count") is not None]
        ),
        "mean_class_minus_object_logit": mean(
            [finite(row.get("class_minus_object_logit")) for row in rows if row.get("class_minus_object_logit") is not None]
        ),
    }


def pair_effects(rows: list[dict[str, Any]], condition: str) -> list[dict[str, Any]]:
    originals = {row_pair_key(row): row for row in rows if row.get("condition_type") == "original"}
    grouped: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for row in rows:
        if row.get("condition_type") != condition:
            continue
        base = originals.get(row_pair_key(row))
        if base:
            grouped[f"{row.get('domain')}::{row.get('candidate_key')}"].append((base, row))
    out: list[dict[str, Any]] = []
    for key, pairs in grouped.items():
        first = pairs[0][1]
        clear_gain = sum(1 for base, row in pairs if not base.get("rollout_clear_answer_class") and row.get("rollout_clear_answer_class"))
        clear_loss = sum(1 for base, row in pairs if base.get("rollout_clear_answer_class") and not row.get("rollout_clear_answer_class"))
        rollout_gain = sum(1 for base, row in pairs if not base.get("rollout_answer_class") and row.get("rollout_answer_class"))
        rollout_loss = sum(1 for base, row in pairs if base.get("rollout_answer_class") and not row.get("rollout_answer_class"))
        first_gain = sum(1 for base, row in pairs if not base.get("first_token_answer_class") and row.get("first_token_answer_class"))
        first_loss = sum(1 for base, row in pairs if base.get("first_token_answer_class") and not row.get("first_token_answer_class"))
        echo_reduced = sum(1 for base, row in pairs if base.get("rollout_object_echo") and not row.get("rollout_object_echo"))
        echo_induced = sum(1 for base, row in pairs if not base.get("rollout_object_echo") and row.get("rollout_object_echo"))
        blocker_reduction = [
            finite(base.get("class_blocker_count")) - finite(row.get("class_blocker_count"))
            for base, row in pairs
            if base.get("class_blocker_count") is not None and row.get("class_blocker_count") is not None
        ]
        margin_gain = [
            finite(row.get("class_minus_object_logit")) - finite(base.get("class_minus_object_logit"))
            for base, row in pairs
            if base.get("class_minus_object_logit") is not None and row.get("class_minus_object_logit") is not None
        ]
        score = (
            3.0 * clear_gain
            + 2.0 * rollout_gain
            + first_gain
            + echo_reduced
            + 0.15 * (mean(blocker_reduction) or 0.0)
            + 0.15 * (mean(margin_gain) or 0.0)
            - 3.0 * clear_loss
            - 2.0 * rollout_loss
            - first_loss
            - echo_induced
        )
        out.append(
            {
                "condition_type": condition,
                "domain": first.get("domain"),
                "candidate_key": first.get("candidate_key"),
                "candidate_role": first.get("candidate_role"),
                "mode": first.get("edit_mode"),
                "gear_keys": first.get("gear_keys"),
                "n_pairs": len(pairs),
                "first_gain": first_gain,
                "first_loss": first_loss,
                "rollout_gain": rollout_gain,
                "rollout_loss": rollout_loss,
                "clear_rollout_gain": clear_gain,
                "clear_rollout_loss": clear_loss,
                "object_echo_reduced": echo_reduced,
                "object_echo_induced": echo_induced,
                "mean_blocker_reduction": mean(blocker_reduction),
                "mean_class_minus_object_gain": mean(margin_gain),
                "effect_score": score,
            }
        )
    out.sort(key=lambda row: finite(row.get("effect_score")), reverse=True)
    return out


def summarize(args: argparse.Namespace, attn_impl: str | None, cases: list[dict[str, Any]], rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_domain: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_condition[str(row.get("condition_type"))].append(row)
        by_domain[str(row.get("domain"))].append(row)
    best = pair_effects(rows, "best_holdout")
    alternate = pair_effects(rows, "alternate_mode")
    control = pair_effects(rows, "same_layer_control")
    shared = pair_effects(rows, "shared_exact_probe")
    return {
        "phase": PHASE,
        "title": "Domain Gear Holdout and Sign Calibration Audit",
        "model": args.model,
        "round": args.round_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "attn_implementation": attn_impl,
        "dtype": "bfloat16",
        "quantization": "off",
        "source_phase858_round": args.source_round,
        "n_cases": len(cases),
        "domains": sorted({str(case.get("domain")) for case in cases}),
        "prompt_variants": parse_csv(args.prompt_variants),
        "n_rows": len(rows),
        "condition_summary": {key: compact(group) for key, group in sorted(by_condition.items())},
        "domain_summary": {key: compact(group) for key, group in sorted(by_domain.items())},
        "best_holdout_effects": best,
        "alternate_mode_effects": alternate,
        "same_layer_control_effects": control,
        "shared_exact_probe_effects": shared,
        "calibration_summary": {
            "best_domains_positive": sum(1 for row in best if finite(row.get("effect_score")) > 0),
            "best_domains_clear_gain": sum(1 for row in best if int(row.get("clear_rollout_gain") or 0) > 0),
            "alternate_domains_clear_gain": sum(1 for row in alternate if int(row.get("clear_rollout_gain") or 0) > 0),
            "control_domains_clear_gain": sum(1 for row in control if int(row.get("clear_rollout_gain") or 0) > 0),
            "shared_probe_domains_clear_gain": sum(1 for row in shared if int(row.get("clear_rollout_gain") or 0) > 0),
        },
        "boundary": (
            "This phase validates Phase 858 top domain gears on held-out objects and compares best mode, alternate mode, "
            "and same-layer controls. It is sign calibration and holdout audit, not final closure."
        ),
    }


def eval_model(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = RESULT_ROOT / args.round_name
    out_dir.mkdir(parents=True, exist_ok=True)
    source_summary = phase858_summary(args.model, args.source_round)
    candidates = phase858_candidates(args.model, args.source_round)
    used_cases = used_phase858_cases(args.model, args.source_round)
    cases = selected_holdout_cases(args, used_cases)
    prompt_variants = parse_csv(args.prompt_variants)
    by_domain: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        by_domain[str(case.get("domain"))].append(case)
    specs_by_domain = {
        domain: condition_specs_for_domain(domain, source_summary, candidates, args) for domain in by_domain
    }
    if args.dry_run:
        payload = {
            "phase": PHASE,
            "model": args.model,
            "round": args.round_name,
            "source_round": args.source_round,
            "cases": cases,
            "specs": {domain: [{k: v for k, v in spec.items() if k != "gears"} for spec in specs] for domain, specs in specs_by_domain.items()},
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
        return payload

    model = None
    tokenizer = None
    rows: list[dict[str, Any]] = []
    try:
        model, tokenizer, device, attn_impl = p844.p828.p796.load_model_bf16_prefer_flash(
            args.model, args.attn_implementations
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        n_layers = len(get_layers(model))
        for domain_idx, (domain, domain_cases) in enumerate(sorted(by_domain.items()), 1):
            specs = specs_by_domain.get(domain) or []
            for case in domain_cases:
                sets = p856.token_sets(tokenizer, case)
                for prompt_variant in prompt_variants:
                    prompt = p856.prompt_for_case(case, prompt_variant)
                    prompt_ids = p844.encode_prompt(tokenizer, prompt)
                    for spec in specs:
                        valid_gears = [
                            gear
                            for gear in spec.get("gears", [])
                            if 0 <= int(gear["layer_idx"]) < n_layers and int(gear["channel_id"]) >= 0
                        ]
                        logits = p844.first_logits_with_gears(model, device, prompt_ids, valid_gears, str(spec["mode"]))
                        first = p856.first_token_metrics(tokenizer, logits, sets, int(args.topk_tokens))
                        generated, token_ids = p844.greedy_with_gears(
                            model,
                            tokenizer,
                            device,
                            prompt_ids,
                            valid_gears,
                            str(spec["mode"]),
                            int(args.max_new_tokens),
                        )
                        rollout = p856.classify_rollout(generated, case)
                        rows.append(
                            {
                                "row_kind": "phase859_domain_gear_holdout_sign_calibration_audit",
                                "phase": PHASE,
                                "model": args.model,
                                "round": args.round_name,
                                "source_round": args.source_round,
                                "domain": domain,
                                "case_id": case["case_id"],
                                "object": case["object"],
                                "canonical_answer": case["canonical_answer"],
                                "answer_aliases": case["answer_aliases"],
                                "overlap_kind": case["overlap_kind"],
                                "prompt_variant": prompt_variant,
                                "prompt": prompt,
                                "condition_type": spec["condition_type"],
                                "candidate_key": spec["candidate_key"],
                                "candidate_role": spec["candidate_role"],
                                "edit_mode": spec["mode"],
                                "source_domains": spec.get("source_domains"),
                                "source_effect_score": spec.get("source_effect_score"),
                                "gear_count": len(valid_gears),
                                "gear_keys": [gear_key(gear) for gear in valid_gears],
                                "token_ids": token_ids,
                                **sets,
                                **first,
                                **rollout,
                            }
                        )
            log(f"{args.model}/{args.round_name}: holdout domain {domain_idx}/{len(by_domain)} rows={len(rows)}")
    finally:
        if model is not None:
            p844.p828.release_model(model)
        if tokenizer is not None:
            del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary = summarize(args, attn_impl, cases, rows)
    p846.write_jsonl(out_dir / f"phase859_{args.model}_rows.jsonl", rows)
    p846.write_json(out_dir / f"phase859_{args.model}_summary.json", summary)
    print(
        json.dumps(
            {
                "model": args.model,
                "round": args.round_name,
                "rows": len(rows),
                **summary["calibration_summary"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return summary


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        f"# Phase 859 Domain Gear Holdout and Sign Calibration Audit ({payload['round']})",
        "",
        "- Source: Phase 858 confirm top domain gears.",
        "- Boundary: holdout/sign calibration, not language closure.",
        "",
        "## Cross-Model Summary",
        "",
        "| model | rows | best positive domains | best clear domains | alternate clear domains | control clear domains | shared probe clear domains |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for model_name in MODELS:
        data = payload.get("model_summaries", {}).get(model_name) or {}
        cal = data.get("calibration_summary") or {}
        lines.append(
            f"| {model_name} | {data.get('n_rows', 0)} | {cal.get('best_domains_positive', 0)} | "
            f"{cal.get('best_domains_clear_gain', 0)} | {cal.get('alternate_domains_clear_gain', 0)} | "
            f"{cal.get('control_domains_clear_gain', 0)} | {cal.get('shared_probe_domains_clear_gain', 0)} |"
        )
    lines += [
        "",
        "## Best Holdout Effects",
        "",
        "| model | domain | role | mode | gears | pairs | score | first gain/loss | rollout gain/loss | clear gain/loss | echo reduced/induced | blocker reduction | margin gain |",
        "|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model_name in MODELS:
        data = payload.get("model_summaries", {}).get(model_name) or {}
        for row in data.get("best_holdout_effects") or []:
            lines.append(
                f"| {model_name} | `{row.get('domain')}` | `{row.get('candidate_role')}` | `{row.get('mode')}` | "
                f"`{'+'.join(row.get('gear_keys') or [])}` | {row.get('n_pairs', 0)} | {fmt(row.get('effect_score'))} | "
                f"{row.get('first_gain', 0)}/{row.get('first_loss', 0)} | "
                f"{row.get('rollout_gain', 0)}/{row.get('rollout_loss', 0)} | "
                f"{row.get('clear_rollout_gain', 0)}/{row.get('clear_rollout_loss', 0)} | "
                f"{row.get('object_echo_reduced', 0)}/{row.get('object_echo_induced', 0)} | "
                f"{fmt(row.get('mean_blocker_reduction'))} | {fmt(row.get('mean_class_minus_object_gain'))} |"
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def summarize_round(round_name: str) -> dict[str, Any]:
    out_dir = RESULT_ROOT / round_name
    payload: dict[str, Any] = {
        "phase": PHASE,
        "round": round_name,
        "status": "missing",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "models": [],
        "model_summaries": {},
    }
    for model_name in MODELS:
        path = out_dir / f"phase859_{model_name}_summary.json"
        if path.exists():
            payload["models"].append(model_name)
            payload["model_summaries"][model_name] = read_json(path)
    payload["status"] = "complete" if len(payload["models"]) == len(MODELS) else "partial"
    p846.write_json(out_dir / "phase859_cross_model_summary.json", payload)
    write_markdown(out_dir / "phase859_cross_model_summary.md", payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--round-name", default="holdout")
    parser.add_argument("--source-round", default="confirm")
    parser.add_argument("--domains", default="geometry,animal,tool,color,material,abstract,plant,object")
    parser.add_argument("--max-holdout-cases-per-domain", type=int, default=2)
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--prompt-variants", default="natural_question,natural_category")
    parser.add_argument("--include-shared-exact-probe", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--topk-tokens", type=int, default=20)
    parser.add_argument("--attn-implementations", default="flash_attention_2,sdpa")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summarize-round", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.summarize_round:
        payload = summarize_round(args.round_name)
        print(json.dumps({"phase": PHASE, "round": args.round_name, "status": payload.get("status"), "models": payload.get("models")}, ensure_ascii=False, indent=2), flush=True)
        return
    if not args.model:
        raise SystemExit("--model is required unless --summarize-round is set")
    eval_model(args)


if __name__ == "__main__":
    main()
