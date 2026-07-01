#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))
sys.path.insert(0, str(ROOT / "tests" / "gpt5"))

import phase796_global_competitor_token_identity_audit as p796  # noqa: E402
import phase816_multi_token_answer_span_rollout_closure as p816  # noqa: E402
from model_utils import release_model  # noqa: E402
from phase722_functional_head_atlas_causal_ablation import write_json, write_jsonl  # noqa: E402
from phase735_source_restricted_writer_validation import MODELS  # noqa: E402
from phase776_readout_bridge_competition_audit import normalize_token_text  # noqa: E402


PHASE = 818
RESULT_ROOT = Path("tests/result/phase818_alias_span_candidate_scoring_benchmark")


TARGET_ALIASES: dict[str, list[str]] = {
    "p816_cat_living_thing": ["living thing", "living organism", "animal", "mammal", "domestic animal"],
    "p816_hammer_hand_tool": ["hand tool", "manual tool", "tool"],
    "p816_bus_public_transport": ["public transport", "public transportation", "public transit", "transit vehicle"],
    "p816_guitar_musical_instrument": ["musical instrument", "music instrument", "instrument", "string instrument"],
    "p816_rose_flowering_plant": ["flowering plant", "flower", "flowers", "flowering shrub"],
    "p816_chair_household_furniture": ["household furniture", "furniture", "seating furniture"],
    "p816_apple_edible_fruit": ["edible fruit", "fruit"],
    "p816_heart_body_organ": ["body organ", "organ"],
    "p816_red_warm_color": ["warm color", "warm colour", "warm hue", "red color", "red colour"],
    "p816_salmon_aquatic_animal": ["aquatic animal", "fish", "salmon fish"],
    "p816_oak_tall_tree": ["tall tree", "tree", "hardwood tree", "tree species"],
    "p816_carrot_root_vegetable": ["root vegetable", "vegetable"],
    "p816_laptop_electronic_device": ["electronic device", "computer", "personal computer", "computing device"],
    "p816_spoon_eating_utensil": ["eating utensil", "kitchen utensil", "utensil", "cutlery"],
    "p816_triangle_geometric_shape": ["geometric shape", "geometry shape", "shape", "polygon"],
    "p816_winter_cold_season": ["cold season", "winter season", "season"],
    "p816_gold_precious_metal": ["precious metal", "valuable metal"],
    "p816_oxygen_chemical_element": ["chemical element", "element"],
    "p816_cactus_desert_plant": ["desert plant", "cactus plant", "cactus", "succulent plant"],
    "p816_doctor_medical_worker": ["medical worker", "medical professional", "health professional", "healthcare worker"],
}


NEAR_MISSES: dict[str, list[str]] = {
    "p816_cat_living_thing": ["pet", "pet category", "organism", "creature"],
    "p816_hammer_hand_tool": ["hardware", "implement", "instrument"],
    "p816_bus_public_transport": ["vehicle", "transport vehicle", "transportation"],
    "p816_guitar_musical_instrument": ["object", "tool", "sound device"],
    "p816_rose_flowering_plant": ["plant", "flower category", "ornamental plant"],
    "p816_chair_household_furniture": ["household object", "object", "seat"],
    "p816_apple_edible_fruit": ["food", "plant product", "produce"],
    "p816_heart_body_organ": ["body part", "human body part", "anatomy"],
    "p816_red_warm_color": ["color", "colour", "hue", "color category"],
    "p816_salmon_aquatic_animal": ["animal", "seafood", "food"],
    "p816_oak_tall_tree": ["plant", "wood", "forest plant"],
    "p816_carrot_root_vegetable": ["food", "edible plant", "plant"],
    "p816_laptop_electronic_device": ["device", "electronics", "machine"],
    "p816_spoon_eating_utensil": ["tool", "kitchen tool", "household object"],
    "p816_triangle_geometric_shape": ["geometry", "mathematical object", "figure"],
    "p816_winter_cold_season": ["cold weather", "weather", "time period"],
    "p816_gold_precious_metal": ["metal", "material", "element"],
    "p816_oxygen_chemical_element": ["gas", "chemical", "substance"],
    "p816_cactus_desert_plant": ["plant", "succulent", "desert organism"],
    "p816_doctor_medical_worker": ["worker", "person", "profession", "occupation"],
}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def norm_text(value: Any) -> str:
    text = normalize_token_text("" if value is None else str(value)).strip().lower()
    for ch in ['"', "'", "`", ".", ",", ";", ":", "[", "]", "(", ")", "{", "}"]:
        text = text.replace(ch, " ")
    return " ".join(text.split())


def parse_csv(text: str) -> list[str]:
    return [x.strip() for x in str(text or "").split(",") if x.strip()]


def unique_phrases(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        raw = str(value).strip()
        key = norm_text(raw)
        if not raw or key in seen:
            continue
        seen.add(key)
        out.append(raw)
    return out


def phrase_variants(phrase: str) -> list[str]:
    return p816.phrase_variants(phrase)


def case_target_aliases(case: dict[str, Any]) -> list[str]:
    aliases = TARGET_ALIASES.get(case["case_id"], [case["answer"]])
    return unique_phrases([case["answer"], *aliases])


def case_near_misses(case: dict[str, Any]) -> list[str]:
    target_norms = {norm_text(x) for x in case_target_aliases(case)}
    values = [x for x in NEAR_MISSES.get(case["case_id"], []) if norm_text(x) not in target_norms]
    return unique_phrases(values)


def span_candidates(tokenizer, case: dict[str, Any], args: argparse.Namespace) -> list[dict[str, Any]]:
    exact_answer = str(case["answer"]).strip()
    exact_norm = norm_text(exact_answer)
    specs: list[dict[str, Any]] = []
    specs.append({"candidate_class": "target_exact", "phrase": exact_answer, "is_exact_target": True, "is_target_alias": True})
    for phrase in case_target_aliases(case):
        if norm_text(phrase) == exact_norm:
            continue
        specs.append({"candidate_class": "target_alias", "phrase": phrase, "is_exact_target": False, "is_target_alias": True})
    for phrase in case_near_misses(case):
        specs.append({"candidate_class": "near_miss", "phrase": phrase, "is_exact_target": False, "is_target_alias": False})
    wrong_phrases = unique_phrases([case["contrast_answer"], *case.get("distractors", [])])
    for phrase in wrong_phrases:
        specs.append({"candidate_class": "wrong", "phrase": phrase, "is_exact_target": False, "is_target_alias": False})
    for phrase in p816.GENERIC_BLOCKERS:
        specs.append({"candidate_class": "generic_blocker", "phrase": phrase, "is_exact_target": False, "is_target_alias": False})

    out: list[dict[str, Any]] = []
    seen: set[tuple[str, tuple[int, ...]]] = set()
    for spec in specs:
        for text in phrase_variants(spec["phrase"]):
            ids = tokenizer.encode(text, add_special_tokens=False)
            if not ids:
                continue
            key = (str(spec["candidate_class"]), tuple(int(x) for x in ids))
            if key in seen:
                continue
            seen.add(key)
            out.append(
                {
                    **spec,
                    "variant_text": text,
                    "token_ids": [int(x) for x in ids],
                    "span_len": len(ids),
                    "normalized_text": norm_text(text),
                }
            )
    return out[: int(args.max_span_candidates)]


def finite(value: Any, default: float = 0.0) -> float:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return default
    return val if math.isfinite(val) else default


def best(scored: list[dict[str, Any]], pred) -> dict[str, Any] | None:
    vals = [row for row in scored if pred(row)]
    return vals[0] if vals else None


def compact_span(row: dict[str, Any] | None) -> dict[str, Any] | None:
    if not row:
        return None
    keep = {
        "candidate_class",
        "phrase",
        "variant_text",
        "is_exact_target",
        "is_target_alias",
        "token_ids",
        "span_len",
        "score_sum_logprob",
        "score_mean_logprob",
        "score_sum_logit",
        "score_mean_logit",
        "step_all_top1",
        "max_rank_above",
    }
    out = {k: row.get(k) for k in keep}
    out["token_logs"] = row.get("token_logs", [])[:3]
    return out


def class_match(generated: Any, phrases: list[str]) -> tuple[bool, str | None, str]:
    gen = norm_text(generated)
    if not gen:
        return False, None, "empty"
    for phrase in phrases:
        pnorm = norm_text(phrase)
        if gen == pnorm:
            return True, phrase, "exact"
    for phrase in phrases:
        pnorm = norm_text(phrase)
        # Generated text may continue after a canonical answer phrase, but a one-word
        # abbreviation should not be treated as a full multi-word alias.
        if gen.startswith(pnorm + " "):
            return True, phrase, "generated_extends_alias"
        if pnorm.startswith(gen + " ") and len(gen.split()) >= 2:
            return True, phrase, "generated_short_alias"
    return False, None, "no_match"


def generation_class(generated: str, case: dict[str, Any]) -> dict[str, Any]:
    clean = p816.clean_generated(generated)
    target_aliases = case_target_aliases(case)
    near = case_near_misses(case)
    wrong = unique_phrases([case["contrast_answer"], *case.get("distractors", [])])
    generic = p816.GENERIC_BLOCKERS
    target_ok, target_phrase, target_kind = class_match(clean, target_aliases)
    near_ok, near_phrase, near_kind = class_match(clean, near)
    wrong_ok, wrong_phrase, wrong_kind = class_match(clean, wrong)
    generic_ok, generic_phrase, generic_kind = class_match(clean, generic)
    if target_ok:
        cls = "target_alias"
        phrase = target_phrase
        kind = target_kind
    elif near_ok:
        cls = "near_miss"
        phrase = near_phrase
        kind = near_kind
    elif wrong_ok:
        cls = "wrong"
        phrase = wrong_phrase
        kind = wrong_kind
    elif generic_ok:
        cls = "generic_blocker"
        phrase = generic_phrase
        kind = generic_kind
    elif not norm_text(clean):
        cls = "empty_or_format"
        phrase = None
        kind = "empty"
    else:
        cls = "other"
        phrase = None
        kind = "no_match"
    exact_ok, exact_phrase, exact_kind = class_match(clean, [case["answer"]])
    return {
        "generated_text": generated,
        "generated_clean": clean,
        "generated_norm": norm_text(clean),
        "generation_class": cls,
        "generation_matched_phrase": phrase,
        "generation_match_kind": kind,
        "generation_exact_target": exact_ok,
        "generation_exact_phrase": exact_phrase,
        "generation_exact_match_kind": exact_kind,
        "generation_target_alias": target_ok,
        "generation_near_miss": near_ok,
        "generation_wrong": wrong_ok,
        "generation_generic_blocker": generic_ok,
    }


def margin(lhs: dict[str, Any] | None, rhs: dict[str, Any] | None) -> float | None:
    if not lhs or not rhs:
        return None
    return finite(lhs.get("score_mean_logprob")) - finite(rhs.get("score_mean_logprob"))


def audit_case(model, tokenizer, device: torch.device, case: dict[str, Any], prompt_variant: str, args: argparse.Namespace) -> dict[str, Any]:
    prompt = p816.build_prompt(case, prompt_variant)
    prompt_ids = [int(x) for x in tokenizer.encode(prompt, add_special_tokens=False)]
    candidates = span_candidates(tokenizer, case, args)
    scored = p816.score_candidates(model, tokenizer, device, prompt_ids, candidates, args.batch_size, args.top_k)

    best_exact = best(scored, lambda row: bool(row.get("is_exact_target")))
    best_target_alias = best(scored, lambda row: bool(row.get("is_target_alias")))
    best_non_exact = best(scored, lambda row: not bool(row.get("is_exact_target")))
    best_non_alias = best(scored, lambda row: not bool(row.get("is_target_alias")))
    best_near = best(scored, lambda row: row.get("candidate_class") == "near_miss")
    best_wrong = best(scored, lambda row: row.get("candidate_class") == "wrong")
    best_generic = best(scored, lambda row: row.get("candidate_class") == "generic_blocker")
    generated, generated_ids = p816.greedy_generate(model, tokenizer, device, prompt_ids, args.max_new_tokens)
    gen = generation_class(generated, case)

    exact_margin_vs_non_exact = margin(best_exact, best_non_exact)
    alias_margin_vs_non_alias = margin(best_target_alias, best_non_alias)
    alias_margin_vs_near = margin(best_target_alias, best_near)
    alias_margin_vs_wrong = margin(best_target_alias, best_wrong)
    alias_margin_vs_generic = margin(best_target_alias, best_generic)
    exact_span_score_closure = bool(best_exact and best_non_exact and finite(exact_margin_vs_non_exact) > 0)
    alias_span_score_closure = bool(best_target_alias and best_non_alias and finite(alias_margin_vs_non_alias) > 0)
    near_miss_cleared = bool(best_target_alias and best_near and finite(alias_margin_vs_near) > 0)
    wrong_cleared = bool(best_target_alias and best_wrong and finite(alias_margin_vs_wrong) > 0)
    generic_cleared = bool(best_target_alias and best_generic and finite(alias_margin_vs_generic) > 0)
    exact_rollout = bool(gen["generation_exact_target"])
    alias_rollout = bool(gen["generation_target_alias"])
    exact_full = exact_span_score_closure and exact_rollout and near_miss_cleared and wrong_cleared and generic_cleared
    alias_full = alias_span_score_closure and alias_rollout and near_miss_cleared and wrong_cleared and generic_cleared

    if alias_full and not exact_full:
        label = "alias_class_closes_exact_phrase_fails"
    elif alias_full:
        label = "alias_score_and_rollout_closed"
    elif alias_span_score_closure and not alias_rollout:
        label = "alias_score_closed_rollout_not_closed"
    elif alias_rollout and not alias_span_score_closure:
        label = "alias_rollout_closed_score_not_closed"
    elif best_non_alias and best_non_alias.get("candidate_class") == "near_miss":
        label = "near_miss_span_wins"
    elif best_non_alias and best_non_alias.get("candidate_class") == "wrong":
        label = "wrong_span_wins"
    elif best_non_alias and best_non_alias.get("candidate_class") == "generic_blocker":
        label = "generic_blocker_span_wins"
    else:
        label = "alias_unclosed_other"

    return {
        "row_kind": "phase818_alias_span_candidate_scoring_benchmark",
        "phase": PHASE,
        "model": args.model,
        "round": args.round_name,
        "case_id": case["case_id"],
        "object": case["object"],
        "prompt_variant": prompt_variant,
        "prompt": prompt,
        "target_answer": case["answer"],
        "target_aliases": case_target_aliases(case),
        "near_misses": case_near_misses(case),
        "wrong_phrases": unique_phrases([case["contrast_answer"], *case.get("distractors", [])]),
        "generic_blockers": p816.GENERIC_BLOCKERS,
        "n_candidates": len(candidates),
        "best_exact_target": compact_span(best_exact),
        "best_target_alias_class": compact_span(best_target_alias),
        "best_non_exact": compact_span(best_non_exact),
        "best_non_alias": compact_span(best_non_alias),
        "best_near_miss": compact_span(best_near),
        "best_wrong": compact_span(best_wrong),
        "best_generic_blocker": compact_span(best_generic),
        "top_scored_spans": [compact_span(x) for x in scored[: int(args.saved_top_spans)]],
        "exact_margin_vs_non_exact_mean_logprob": exact_margin_vs_non_exact,
        "alias_margin_vs_non_alias_mean_logprob": alias_margin_vs_non_alias,
        "alias_margin_vs_near_miss_mean_logprob": alias_margin_vs_near,
        "alias_margin_vs_wrong_mean_logprob": alias_margin_vs_wrong,
        "alias_margin_vs_generic_mean_logprob": alias_margin_vs_generic,
        "exact_span_score_closure": exact_span_score_closure,
        "alias_span_score_closure": alias_span_score_closure,
        "near_miss_cleared": near_miss_cleared,
        "wrong_cleared": wrong_cleared,
        "generic_blocker_cleared": generic_cleared,
        "strict_alias_step_top1": bool(best_target_alias and best_target_alias.get("step_all_top1")),
        "generated_token_ids": generated_ids,
        **gen,
        "exact_rollout_closure": exact_rollout,
        "alias_rollout_closure": alias_rollout,
        "exact_full_closure": exact_full,
        "alias_full_closure": alias_full,
        "phase818_label": label,
    }


def summarize(rows: list[dict[str, Any]], args: argparse.Namespace, attn_impl: str) -> dict[str, Any]:
    by_variant: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_variant[str(row.get("prompt_variant"))].append(row)
    variant_summary = {}
    for variant, vals in by_variant.items():
        variant_summary[variant] = {
            "n": len(vals),
            "exact_span_score_rows": sum(1 for row in vals if row.get("exact_span_score_closure")),
            "alias_span_score_rows": sum(1 for row in vals if row.get("alias_span_score_closure")),
            "exact_rollout_rows": sum(1 for row in vals if row.get("exact_rollout_closure")),
            "alias_rollout_rows": sum(1 for row in vals if row.get("alias_rollout_closure")),
            "exact_full_rows": sum(1 for row in vals if row.get("exact_full_closure")),
            "alias_full_rows": sum(1 for row in vals if row.get("alias_full_closure")),
            "near_miss_cleared_rows": sum(1 for row in vals if row.get("near_miss_cleared")),
            "wrong_cleared_rows": sum(1 for row in vals if row.get("wrong_cleared")),
            "generic_blocker_cleared_rows": sum(1 for row in vals if row.get("generic_blocker_cleared")),
            "by_generation_class": dict(Counter(row.get("generation_class") for row in vals)),
            "by_label": dict(Counter(row.get("phase818_label") for row in vals)),
        }
    labels = Counter(row.get("phase818_label") for row in rows)
    failure_rows = [
        row
        for row in rows
        if not row.get("alias_full_closure")
    ][:80]
    return {
        "phase": PHASE,
        "title": "Alias Span Candidate Scoring Benchmark",
        "model": args.model,
        "round": args.round_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "attn_implementation": attn_impl,
        "dtype": "bfloat16",
        "quantization": "off",
        "n_rows": len(rows),
        "n_cases": len({row["case_id"] for row in rows}),
        "prompt_variants": parse_csv(args.prompt_variants),
        "exact_span_score_rows": sum(1 for row in rows if row.get("exact_span_score_closure")),
        "alias_span_score_rows": sum(1 for row in rows if row.get("alias_span_score_closure")),
        "exact_rollout_rows": sum(1 for row in rows if row.get("exact_rollout_closure")),
        "alias_rollout_rows": sum(1 for row in rows if row.get("alias_rollout_closure")),
        "exact_full_rows": sum(1 for row in rows if row.get("exact_full_closure")),
        "alias_full_rows": sum(1 for row in rows if row.get("alias_full_closure")),
        "alias_gain_span_score": sum(1 for row in rows if row.get("alias_span_score_closure"))
        - sum(1 for row in rows if row.get("exact_span_score_closure")),
        "alias_gain_rollout": sum(1 for row in rows if row.get("alias_rollout_closure"))
        - sum(1 for row in rows if row.get("exact_rollout_closure")),
        "alias_gain_full": sum(1 for row in rows if row.get("alias_full_closure"))
        - sum(1 for row in rows if row.get("exact_full_closure")),
        "near_miss_cleared_rows": sum(1 for row in rows if row.get("near_miss_cleared")),
        "wrong_cleared_rows": sum(1 for row in rows if row.get("wrong_cleared")),
        "generic_blocker_cleared_rows": sum(1 for row in rows if row.get("generic_blocker_cleared")),
        "strict_alias_step_top1_rows": sum(1 for row in rows if row.get("strict_alias_step_top1")),
        "by_generation_class": dict(Counter(row.get("generation_class") for row in rows)),
        "by_label": dict(labels),
        "by_prompt_variant": variant_summary,
        "failure_rows": failure_rows,
        "boundary": (
            "This phase moves alias handling into teacher-forced candidate scoring and separates target aliases, near-misses, wrong spans, and generic blockers."
        ),
    }


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        f"# Phase 818 Alias Span Candidate Scoring Benchmark ({payload['round']})",
        "",
        "- Boundary: target answer is evaluated as an alias class, while near-miss, wrong, and generic spans remain explicit competitors.",
        "",
        "## Model Summary",
        "",
        "| model | rows | exact score | alias score | exact rollout | alias rollout | exact full | alias full | near cleared | wrong cleared | generic cleared | labels |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for model_name in MODELS:
        data = payload.get("model_summaries", {}).get(model_name)
        if not data:
            continue
        lines.append(
            f"| {model_name} | {data.get('n_rows')} | {data.get('exact_span_score_rows')} | "
            f"{data.get('alias_span_score_rows')} | {data.get('exact_rollout_rows')} | "
            f"{data.get('alias_rollout_rows')} | {data.get('exact_full_rows')} | "
            f"{data.get('alias_full_rows')} | {data.get('near_miss_cleared_rows')} | "
            f"{data.get('wrong_cleared_rows')} | {data.get('generic_blocker_cleared_rows')} | "
            f"`{json.dumps(data.get('by_label') or {}, ensure_ascii=False)}` |"
        )
    lines += ["", "## Prompt Variant Summary", ""]
    lines += [
        "| model | prompt | n | alias score | alias rollout | alias full | generation classes | labels |",
        "|---|---|---:|---:|---:|---:|---|---|",
    ]
    for model_name in MODELS:
        data = payload.get("model_summaries", {}).get(model_name)
        if not data:
            continue
        for prompt, row in sorted((data.get("by_prompt_variant") or {}).items()):
            lines.append(
                f"| {model_name} | {prompt} | {row.get('n')} | {row.get('alias_span_score_rows')} | "
                f"{row.get('alias_rollout_rows')} | {row.get('alias_full_rows')} | "
                f"`{json.dumps(row.get('by_generation_class') or {}, ensure_ascii=False)}` | "
                f"`{json.dumps(row.get('by_label') or {}, ensure_ascii=False)}` |"
            )
    lines += ["", "## First Failure Rows", ""]
    lines += [
        "| model | prompt | case | target | generated | gen class | best alias | best non-alias | margin | label |",
        "|---|---|---|---|---|---|---|---|---:|---|",
    ]
    for model_name in MODELS:
        data = payload.get("model_summaries", {}).get(model_name)
        if not data:
            continue
        for row in data.get("failure_rows", [])[:24]:
            ba = row.get("best_target_alias_class") or {}
            bn = row.get("best_non_alias") or {}
            lines.append(
                f"| {model_name} | {row.get('prompt_variant')} | {row.get('case_id')} | `{row.get('target_answer')}` | "
                f"`{row.get('generated_clean')}` | `{row.get('generation_class')}` | `{ba.get('variant_text')}` | "
                f"`{bn.get('variant_text')}`/{bn.get('candidate_class')} | "
                f"{finite(row.get('alias_margin_vs_non_alias_mean_logprob')):.3f} | `{row.get('phase818_label')}` |"
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_model(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = RESULT_ROOT / args.round_name
    out_dir.mkdir(parents=True, exist_ok=True)
    selected = [p816.CASES[i] for i in p816.select_evenly(len(p816.CASES), int(args.max_cases))]
    prompt_variants = parse_csv(args.prompt_variants)
    log(
        f"{args.model}/{args.round_name}: cases={len(selected)} prompt_variants={prompt_variants} "
        f"batch={args.batch_size} max_spans={args.max_span_candidates}"
    )
    if args.dry_run:
        return {"model": args.model, "selected_cases": [case["case_id"] for case in selected]}
    model, tokenizer, device, attn_impl = p796.load_model_bf16_prefer_flash(args.model, args.attn_implementations)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    rows: list[dict[str, Any]] = []
    try:
        for ci, case in enumerate(selected, 1):
            for prompt_variant in prompt_variants:
                rows.append(audit_case(model, tokenizer, device, case, prompt_variant, args))
            if ci % int(args.log_every) == 0 or ci == len(selected):
                log(f"{args.model}: alias span scoring {ci}/{len(selected)} cases; rows={len(rows)}")
    finally:
        release_model(model)
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    summary = summarize(rows, args, attn_impl)
    write_jsonl(out_dir / f"phase818_{args.model}_rows.jsonl", rows)
    write_json(out_dir / f"phase818_{args.model}_summary.json", summary)
    print(
        json.dumps(
            {
                "model": args.model,
                "round": args.round_name,
                "attn": attn_impl,
                "n_rows": summary["n_rows"],
                "alias_span_score_rows": summary["alias_span_score_rows"],
                "alias_rollout_rows": summary["alias_rollout_rows"],
                "alias_full_rows": summary["alias_full_rows"],
                "by_label": summary["by_label"],
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
        "status": "missing",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_summaries": {},
        "models": [],
    }
    for model_name in MODELS:
        path = out_dir / f"phase818_{model_name}_summary.json"
        if path.exists():
            payload["model_summaries"][model_name] = json.loads(path.read_text(encoding="utf-8"))
            payload["models"].append(model_name)
    payload["status"] = "complete" if len(payload["models"]) == len(MODELS) else "partial"
    write_json(out_dir / "phase818_cross_model_summary.json", payload)
    write_markdown(out_dir / "phase818_cross_model_summary.md", payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--round-name", default="smoke")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-cases", type=int, default=4)
    parser.add_argument("--prompt-variants", default="exact_choices")
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--max-span-candidates", type=int, default=160)
    parser.add_argument("--saved-top-spans", type=int, default=24)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--attn-implementations", default="flash_attention_2,sdpa,eager")
    parser.add_argument("--log-every", type=int, default=1)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.summarize_only:
        payload = summarize_round(args.round_name)
        print(json.dumps({"round": args.round_name, "status": payload["status"], "models": payload["models"]}, ensure_ascii=False, indent=2))
        return
    if not args.model:
        raise SystemExit("--model is required unless --summarize-only")
    result = run_model(args)
    if args.dry_run:
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
