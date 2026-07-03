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
import phase859_domain_gear_holdout_sign_calibration_audit as p859  # noqa: E402
from model_utils import get_layers  # noqa: E402


PHASE = 860
MODELS = p846.MODELS
RESULT_ROOT = Path("tests/result/phase860_replicated_domain_gear_evidence_ladder")
PHASE858_ROOT = Path("tests/result/phase858_cross_domain_independent_gear_isomorphism_audit")
PHASE859_ROOT = Path("tests/result/phase859_domain_gear_holdout_sign_calibration_audit")


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


def gear_key(gear: dict[str, Any]) -> str:
    return p854.gear_key(gear)


def phase858_summary(model_name: str, round_name: str) -> dict[str, Any]:
    return read_json(PHASE858_ROOT / round_name / f"phase858_{model_name}_summary.json")


def phase858_candidates(model_name: str, round_name: str) -> list[dict[str, Any]]:
    return read_jsonl(PHASE858_ROOT / round_name / f"phase858_{model_name}_candidate_gears.jsonl")


def phase858_used_cases(model_name: str, round_name: str) -> set[str]:
    path = PHASE858_ROOT / round_name / f"phase858_{model_name}_rows.jsonl"
    if not path.exists():
        return set()
    return {str(row.get("case_id")) for row in read_jsonl(path)}


def phase859_summary(model_name: str, round_name: str) -> dict[str, Any]:
    return read_json(PHASE859_ROOT / round_name / f"phase859_{model_name}_summary.json")


def phase859_used_cases(model_name: str, round_name: str) -> set[str]:
    path = PHASE859_ROOT / round_name / f"phase859_{model_name}_rows.jsonl"
    if not path.exists():
        return set()
    return {str(row.get("case_id")) for row in read_jsonl(path)}


def target_domains_from_phase859(model_name: str, round_name: str, min_clear_gain: int) -> list[str]:
    summary = phase859_summary(model_name, round_name)
    domains = [
        str(row.get("domain"))
        for row in summary.get("best_holdout_effects") or []
        if int(row.get("clear_rollout_gain") or 0) >= int(min_clear_gain)
    ]
    return sorted(dict.fromkeys(domains))


def selected_cases(args: argparse.Namespace, target_domains: list[str]) -> list[dict[str, Any]]:
    allowed = set(parse_csv(args.domains) or target_domains)
    rows = [dict(case) for case in p856.base_cases() if str(case.get("domain")) in allowed]
    if int(args.max_cases_per_domain) > 0:
        out: list[dict[str, Any]] = []
        counts: Counter[str] = Counter()
        for case in rows:
            domain = str(case.get("domain"))
            if counts[domain] >= int(args.max_cases_per_domain):
                continue
            out.append(case)
            counts[domain] += 1
        rows = out
    return rows


def split_for_case(case_id: str, phase858_cases: set[str], phase859_cases: set[str]) -> str:
    if case_id in phase858_cases:
        return "phase858_seen"
    if case_id in phase859_cases:
        return "phase859_holdout_seen"
    return "new_replication"


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
        splits = defaultdict(list)
        prompts = defaultdict(list)
        for base, row in pairs:
            splits[str(row.get("case_split"))].append((base, row))
            prompts[str(row.get("prompt_variant"))].append((base, row))

        def gain(pair_rows: list[tuple[dict[str, Any], dict[str, Any]]], metric: str) -> int:
            return sum(1 for base, row in pair_rows if not base.get(metric) and row.get(metric))

        def loss(pair_rows: list[tuple[dict[str, Any], dict[str, Any]]], metric: str) -> int:
            return sum(1 for base, row in pair_rows if base.get(metric) and not row.get(metric))

        clear_gain = gain(pairs, "rollout_clear_answer_class")
        clear_loss = loss(pairs, "rollout_clear_answer_class")
        rollout_gain = gain(pairs, "rollout_answer_class")
        rollout_loss = loss(pairs, "rollout_answer_class")
        first_gain = gain(pairs, "first_token_answer_class")
        first_loss = loss(pairs, "first_token_answer_class")
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
        split_clear_gain = {split: gain(split_pairs, "rollout_clear_answer_class") for split, split_pairs in splits.items()}
        prompt_clear_gain = {prompt: gain(prompt_pairs, "rollout_clear_answer_class") for prompt, prompt_pairs in prompts.items()}
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
                "n_cases": len({row.get("case_id") for _, row in pairs}),
                "n_prompts": len({row.get("prompt_variant") for _, row in pairs}),
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
                "split_clear_gain": split_clear_gain,
                "prompt_clear_gain": prompt_clear_gain,
                "effect_score": score,
            }
        )
    out.sort(key=lambda row: finite(row.get("effect_score")), reverse=True)
    return out


def evidence_level(best: dict[str, Any] | None, alternate: dict[str, Any] | None, control: dict[str, Any] | None) -> dict[str, Any]:
    if not best:
        return {"level": 0, "label": "no_replicated_candidate", "reasons": ["no best effect row"]}
    reasons: list[str] = ["phase858_candidate", "phase859_holdout_source"]
    level = 3
    label = "domain_local_holdout_source"
    best_clear = int(best.get("clear_rollout_gain") or 0)
    best_loss = int(best.get("clear_rollout_loss") or 0)
    control_clear = int((control or {}).get("clear_rollout_gain") or 0)
    splits = best.get("split_clear_gain") or {}
    prompts = best.get("prompt_clear_gain") or {}
    split_hits = sum(1 for value in splits.values() if int(value) > 0)
    prompt_hits = sum(1 for value in prompts.values() if int(value) > 0)
    if best_clear > 0 and best_loss == 0:
        level = 4
        label = "replicated_domain_edge"
        reasons.append("phase860_clear_gain_no_loss")
    if best_clear >= 2 and best_loss == 0 and control_clear == 0:
        level = 5
        label = "replicated_control_filtered_domain_gear"
        reasons.append("same_layer_control_clear_zero")
    if best_clear >= 2 and best_loss == 0 and control_clear == 0 and split_hits >= 2 and prompt_hits >= 2:
        level = 5
        label = "multi_split_prompt_replicated_domain_gear"
        reasons.append("multi_split_and_multi_prompt_support")
    if best_clear >= 3 and best_loss == 0 and control_clear == 0 and split_hits >= 2 and prompt_hits >= 3:
        level = 6
        label = "strong_domain_invariant_candidate"
        reasons.append("broad_prompt_replication")
    alternate_clear = int((alternate or {}).get("clear_rollout_gain") or 0)
    if alternate_clear > 0:
        reasons.append("alternate_mode_has_clear_gain_sign_ambiguous")
    return {
        "level": level,
        "label": label,
        "reasons": reasons,
        "best_clear_gain": best_clear,
        "best_clear_loss": best_loss,
        "alternate_clear_gain": alternate_clear,
        "control_clear_gain": control_clear,
        "split_hits": split_hits,
        "prompt_hits": prompt_hits,
    }


def summarize(args: argparse.Namespace, attn_impl: str | None, cases: list[dict[str, Any]], rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_domain: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_condition[str(row.get("condition_type"))].append(row)
        by_domain[str(row.get("domain"))].append(row)
        by_split[str(row.get("case_split"))].append(row)
    best = pair_effects(rows, "best_holdout")
    alternate = pair_effects(rows, "alternate_mode")
    control = pair_effects(rows, "same_layer_control")
    best_by_domain = {str(row.get("domain")): row for row in best}
    alt_by_domain = {str(row.get("domain")): row for row in alternate}
    ctl_by_domain = {str(row.get("domain")): row for row in control}
    ladder = {
        domain: evidence_level(best_by_domain.get(domain), alt_by_domain.get(domain), ctl_by_domain.get(domain))
        for domain in sorted({str(case.get("domain")) for case in cases})
    }
    return {
        "phase": PHASE,
        "title": "Replicated Domain Gear Evidence Ladder",
        "model": args.model,
        "round": args.round_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "attn_implementation": attn_impl,
        "dtype": "bfloat16",
        "quantization": "off",
        "source_phase858_round": args.source_phase858_round,
        "source_phase859_round": args.source_phase859_round,
        "target_domains": sorted({str(case.get("domain")) for case in cases}),
        "n_cases": len(cases),
        "prompt_variants": parse_csv(args.prompt_variants),
        "n_rows": len(rows),
        "condition_summary": {key: compact(group) for key, group in sorted(by_condition.items())},
        "domain_summary": {key: compact(group) for key, group in sorted(by_domain.items())},
        "split_summary": {key: compact(group) for key, group in sorted(by_split.items())},
        "best_effects": best,
        "alternate_effects": alternate,
        "same_layer_control_effects": control,
        "evidence_ladder": ladder,
        "ladder_summary": {
            "max_level": max((int(row.get("level", 0)) for row in ladder.values()), default=0),
            "domains_level_4_plus": sum(1 for row in ladder.values() if int(row.get("level", 0)) >= 4),
            "domains_level_5_plus": sum(1 for row in ladder.values() if int(row.get("level", 0)) >= 5),
            "domains_level_6": sum(1 for row in ladder.values() if int(row.get("level", 0)) >= 6),
            "domains_with_sign_ambiguity": sum(
                1 for row in ladder.values() if "alternate_mode_has_clear_gain_sign_ambiguous" in row.get("reasons", [])
            ),
        },
        "boundary": (
            "This phase replicates Phase 859 positive domain gears across all available base cases and expanded prompt gates. "
            "It builds a domain evidence ladder; it does not prove cross-domain universal language closure."
        ),
    }


def eval_model(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = RESULT_ROOT / args.round_name
    out_dir.mkdir(parents=True, exist_ok=True)
    source858 = phase858_summary(args.model, args.source_phase858_round)
    candidates858 = phase858_candidates(args.model, args.source_phase858_round)
    target_domains = parse_csv(args.domains) or target_domains_from_phase859(
        args.model, args.source_phase859_round, int(args.min_phase859_clear_gain)
    )
    cases = selected_cases(args, target_domains)
    phase858_cases = phase858_used_cases(args.model, args.source_phase858_round)
    phase859_cases = phase859_used_cases(args.model, args.source_phase859_round)
    prompt_variants = parse_csv(args.prompt_variants)
    by_domain: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        by_domain[str(case.get("domain"))].append(case)
    specs_by_domain = {
        domain: p859.condition_specs_for_domain(domain, source858, candidates858, args) for domain in by_domain
    }
    if args.dry_run:
        payload = {
            "phase": PHASE,
            "model": args.model,
            "round": args.round_name,
            "target_domains": target_domains,
            "cases": cases,
            "prompt_variants": prompt_variants,
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
                case_split = split_for_case(str(case.get("case_id")), phase858_cases, phase859_cases)
                for prompt_variant in prompt_variants:
                    prompt = p856.prompt_for_case(case, prompt_variant)
                    prompt_ids = p844.encode_prompt(tokenizer, prompt)
                    for spec in specs:
                        if spec["condition_type"] == "shared_exact_probe":
                            continue
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
                                "row_kind": "phase860_replicated_domain_gear_evidence_ladder",
                                "phase": PHASE,
                                "model": args.model,
                                "round": args.round_name,
                                "domain": domain,
                                "case_id": case["case_id"],
                                "case_split": case_split,
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
                                "source_effect_score": spec.get("source_effect_score"),
                                "gear_count": len(valid_gears),
                                "gear_keys": [gear_key(gear) for gear in valid_gears],
                                "token_ids": token_ids,
                                **sets,
                                **first,
                                **rollout,
                            }
                        )
            log(f"{args.model}/{args.round_name}: replicated domain {domain_idx}/{len(by_domain)} rows={len(rows)}")
    finally:
        if model is not None:
            p844.p828.release_model(model)
        if tokenizer is not None:
            del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary = summarize(args, attn_impl, cases, rows)
    p846.write_jsonl(out_dir / f"phase860_{args.model}_rows.jsonl", rows)
    p846.write_json(out_dir / f"phase860_{args.model}_summary.json", summary)
    print(
        json.dumps(
            {
                "model": args.model,
                "round": args.round_name,
                "rows": len(rows),
                **summary["ladder_summary"],
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
        f"# Phase 860 Replicated Domain Gear Evidence Ladder ({payload['round']})",
        "",
        "- Source: Phase 859 clear-gain domain gears.",
        "- Boundary: replicated domain evidence ladder, not cross-domain language closure.",
        "",
        "## Cross-Model Summary",
        "",
        "| model | rows | max level | L4+ domains | L5+ domains | L6 domains | sign ambiguous domains |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for model_name in MODELS:
        data = payload.get("model_summaries", {}).get(model_name) or {}
        ladder = data.get("ladder_summary") or {}
        lines.append(
            f"| {model_name} | {data.get('n_rows', 0)} | {ladder.get('max_level', 0)} | "
            f"{ladder.get('domains_level_4_plus', 0)} | {ladder.get('domains_level_5_plus', 0)} | "
            f"{ladder.get('domains_level_6', 0)} | {ladder.get('domains_with_sign_ambiguity', 0)} |"
        )
    lines += [
        "",
        "## Evidence Ladder",
        "",
        "| model | domain | level | label | best gears | best clear gain/loss | split hits | prompt hits | alternate clear | control clear | reasons |",
        "|---|---|---:|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for model_name in MODELS:
        data = payload.get("model_summaries", {}).get(model_name) or {}
        best = {str(row.get("domain")): row for row in data.get("best_effects") or []}
        for domain, row in sorted((data.get("evidence_ladder") or {}).items()):
            best_row = best.get(domain) or {}
            lines.append(
                f"| {model_name} | `{domain}` | {row.get('level', 0)} | `{row.get('label')}` | "
                f"`{'+'.join(best_row.get('gear_keys') or [])}` | "
                f"{row.get('best_clear_gain', 0)}/{row.get('best_clear_loss', 0)} | "
                f"{row.get('split_hits', 0)} | {row.get('prompt_hits', 0)} | "
                f"{row.get('alternate_clear_gain', 0)} | {row.get('control_clear_gain', 0)} | "
                f"`{','.join(row.get('reasons') or [])}` |"
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
        path = out_dir / f"phase860_{model_name}_summary.json"
        if path.exists():
            payload["models"].append(model_name)
            payload["model_summaries"][model_name] = read_json(path)
    payload["status"] = "complete" if len(payload["models"]) == len(MODELS) else "partial"
    p846.write_json(out_dir / "phase860_cross_model_summary.json", payload)
    write_markdown(out_dir / "phase860_cross_model_summary.md", payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--round-name", default="replicate")
    parser.add_argument("--source-phase858-round", default="confirm")
    parser.add_argument("--source-phase859-round", default="holdout")
    parser.add_argument("--domains", default="")
    parser.add_argument("--min-phase859-clear-gain", type=int, default=1)
    parser.add_argument("--max-cases-per-domain", type=int, default=5)
    parser.add_argument("--prompt-variants", default="natural_question,natural_category,classification")
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
        print(
            json.dumps(
                {"phase": PHASE, "round": args.round_name, "status": payload.get("status"), "models": payload.get("models")},
                ensure_ascii=False,
                indent=2,
            ),
            flush=True,
        )
        return
    if not args.model:
        raise SystemExit("--model is required unless --summarize-round is set")
    eval_model(args)


if __name__ == "__main__":
    main()
