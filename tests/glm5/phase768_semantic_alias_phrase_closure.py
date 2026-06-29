#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import math
import re
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))
sys.path.insert(0, str(ROOT / "tests" / "gpt5"))

from model_utils import release_model  # noqa: E402
from phase765_commonsense_context_identity_closure_test import (  # noqa: E402
    VALUE_POOLS,
    build_cases,
    prompt_for_case,
)
from phase767_commonsense_failure_type_topk_audit import (  # noqa: E402
    MODELS,
    load_model_bf16,
    value_aliases,
)


OUT_ROOT = Path("results/glm5_phase768_semantic_alias_phrase_closure")
RESULT_ROOT = Path("tests/result/phase768_semantic_alias_phrase_closure")
PHASE767_ROOT = Path("tests/result/phase767_commonsense_failure_type_topk_audit")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def safe_mean(values: list[Any]) -> float | None:
    vals = []
    for value in values:
        try:
            val = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(val):
            vals.append(val)
    return sum(vals) / len(vals) if vals else None


def fmt(value: Any) -> str:
    if value is None:
        return "null"
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return str(value)


def load_phase767_rows(model_name: str, round_name: str) -> dict[str, dict[str, Any]]:
    path = PHASE767_ROOT / round_name / f"phase767_{model_name}_rows.jsonl"
    if not path.exists():
        path = Path("results/glm5_phase767_commonsense_failure_type_topk_audit") / round_name / f"phase767_{model_name}_rows.jsonl"
    rows: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                rows[row["case_id"]] = row
    return rows


def phrase_forms(value: str) -> list[str]:
    forms = []
    for alias in value_aliases(value):
        for form in [alias, f"{alias}."]:
            if form not in forms:
                forms.append(form)
    return forms


def completion_ids(tokenizer, text: str) -> list[int]:
    ids = tokenizer.encode(" " + text, add_special_tokens=False)
    if not ids:
        ids = tokenizer.encode(text, add_special_tokens=False)
    return [int(x) for x in ids]


def score_phrase_candidates(model, tokenizer, device, prompt: str, values: list[str]) -> list[dict[str, Any]]:
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    candidates = []
    for value in values:
        for phrase in phrase_forms(value):
            comp = completion_ids(tokenizer, phrase)
            if comp:
                candidates.append({"value": value, "phrase": phrase, "completion_ids": comp})
    max_len = max(len(prompt_ids) + len(c["completion_ids"]) for c in candidates)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    batch = []
    mask = []
    for cand in candidates:
        ids = prompt_ids + cand["completion_ids"]
        pad_n = max_len - len(ids)
        batch.append(ids + [pad_id] * pad_n)
        mask.append([1] * len(ids) + [0] * pad_n)
    input_ids = torch.tensor(batch, device=device)
    attention_mask = torch.tensor(mask, device=device)
    with torch.inference_mode():
        out = model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True, use_cache=False)
    logits = out.logits.detach().float().cpu()
    scored = []
    start = len(prompt_ids)
    for b, cand in enumerate(candidates):
        token_logprobs = []
        for j, tid in enumerate(cand["completion_ids"]):
            pos = start + j - 1
            lp = torch.log_softmax(logits[b, pos], dim=-1)
            token_logprobs.append(float(lp[int(tid)].item()))
        scored.append(
            {
                **cand,
                "completion_ids": [int(x) for x in cand["completion_ids"]],
                "sum_logprob": float(sum(token_logprobs)),
                "avg_logprob": float(sum(token_logprobs) / len(token_logprobs)),
                "n_tokens": len(token_logprobs),
            }
        )
    return scored


def best_by_value(scored: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best: dict[str, dict[str, Any]] = {}
    for row in scored:
        value = row["value"]
        if value not in best or row["sum_logprob"] > best[value]["sum_logprob"]:
            best[value] = row
    rows = list(best.values())
    rows.sort(key=lambda r: r["sum_logprob"], reverse=True)
    for i, row in enumerate(rows, 1):
        row["phrase_rank"] = i
    return rows


def clean_generated(text: str) -> str:
    text = text.strip()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines:
        text = lines[0]
    text = re.sub(r"^[\s:：\-–—,.;]+", "", text)
    return text.strip()


def parse_generation(generated: str, values: list[str]) -> dict[str, Any]:
    cleaned = clean_generated(generated)
    lower = cleaned.lower()
    first = re.match(r"[A-Za-z_]+", lower)
    first_word = first.group(0) if first else ""
    starts_value = None
    contains_value = None
    for value in values:
        aliases = {a.lower() for a in value_aliases(value)}
        if first_word in aliases:
            starts_value = value
            break
    for value in values:
        aliases = sorted({re.escape(a.lower()) for a in value_aliases(value)}, key=len, reverse=True)
        if any(re.search(rf"\\b{alias}\\b", lower) for alias in aliases):
            contains_value = value
            break
    return {
        "generated_clean": cleaned,
        "generated_first_word": first_word,
        "generated_starts_value": starts_value,
        "generated_contains_value": contains_value,
    }


def generate_short(model, tokenizer, device, prompt: str, max_new_tokens: int) -> str:
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    input_ids = torch.tensor([ids], device=device)
    attention_mask = torch.ones_like(input_ids)
    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "pad_token_id": tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    with torch.inference_mode():
        out = model.generate(input_ids=input_ids, attention_mask=attention_mask, **gen_kwargs)
    new_ids = out[0, input_ids.shape[1] :].detach().cpu().tolist()
    return tokenizer.decode(new_ids, skip_special_tokens=True)


def audit_case(model, tokenizer, device, case: dict[str, Any], phase767: dict[str, Any], max_new_tokens: int) -> dict[str, Any]:
    prompt = prompt_for_case(case)
    values = VALUE_POOLS[case["relation_key"]]
    scored = score_phrase_candidates(model, tokenizer, device, prompt, values)
    ranked = best_by_value(scored)
    target = next(row for row in ranked if row["value"] == case["answer"])
    best_other = next((row for row in ranked if row["value"] != case["answer"]), None)
    contrast = next((row for row in ranked if row["value"] == case["contrast_answer"]), None)
    generated = generate_short(model, tokenizer, device, prompt, max_new_tokens)
    parsed = parse_generation(generated, values)
    phrase_top1 = target["phrase_rank"] == 1
    generation_match = parsed["generated_starts_value"] == case["answer"]
    return {
        "row_kind": "semantic_alias_phrase_closure",
        "case_id": case["case_id"],
        "context_format": case["context_format"],
        "object": case["object"],
        "domain": case["domain"],
        "relation": case["relation"],
        "target_answer": case["answer"],
        "contrast_answer": case["contrast_answer"],
        "phase767_semantic_top1": bool(phase767.get("target_top1")),
        "phase767_exact_top1": bool(phase767.get("exact_target_top1")),
        "phase767_failure_type": phase767.get("failure_type"),
        "phase767_semantic_rank": phase767.get("target_rank"),
        "phase767_exact_rank": phase767.get("exact_target_rank"),
        "phrase_target_rank": target["phrase_rank"],
        "phrase_target_top1": phrase_top1,
        "phrase_target_sum_logprob": target["sum_logprob"],
        "phrase_target_avg_logprob": target["avg_logprob"],
        "phrase_target_best_form": target["phrase"],
        "phrase_best_value": ranked[0]["value"],
        "phrase_best_form": ranked[0]["phrase"],
        "phrase_best_sum_logprob": ranked[0]["sum_logprob"],
        "phrase_contrast_rank": contrast["phrase_rank"] if contrast else None,
        "phrase_margin_vs_best_other": target["sum_logprob"] - best_other["sum_logprob"] if best_other else None,
        "value_phrase_ranking": [
            {
                "value": row["value"],
                "rank": row["phrase_rank"],
                "best_form": row["phrase"],
                "sum_logprob": row["sum_logprob"],
                "avg_logprob": row["avg_logprob"],
                "n_tokens": row["n_tokens"],
            }
            for row in ranked
        ],
        "generated_text": generated,
        "generation_match": generation_match,
        **parsed,
    }


def summarize_rows(model_name: str, round_name: str, rows: list[dict[str, Any]], attn_impl: str) -> dict[str, Any]:
    def summarize(items: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "n": len(items),
            "semantic_top1_rate": safe_mean([1.0 if r["phase767_semantic_top1"] else 0.0 for r in items]),
            "exact_top1_rate": safe_mean([1.0 if r["phase767_exact_top1"] else 0.0 for r in items]),
            "phrase_top1_rate": safe_mean([1.0 if r["phrase_target_top1"] else 0.0 for r in items]),
            "generation_match_rate": safe_mean([1.0 if r["generation_match"] else 0.0 for r in items]),
            "mean_phrase_rank": safe_mean([r["phrase_target_rank"] for r in items]),
            "mean_phrase_margin": safe_mean([r["phrase_margin_vs_best_other"] for r in items]),
        }

    groups: dict[str, list[dict[str, Any]]] = {
        "all": rows,
        "exact_clean": [r for r in rows if r["phase767_exact_top1"]],
        "semantic_clean": [r for r in rows if r["phase767_semantic_top1"]],
        "semantic_only": [r for r in rows if r["phase767_semantic_top1"] and not r["phase767_exact_top1"]],
        "semantic_fail": [r for r in rows if not r["phase767_semantic_top1"]],
        "rank_le2": [r for r in rows if int(r.get("phase767_semantic_rank") or 9999) <= 2],
    }
    by_relation = []
    relation_groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        subset = "semantic_clean" if row["phase767_semantic_top1"] else "semantic_fail"
        relation_groups[(row["relation"], subset)].append(row)
    for (relation, subset), items in sorted(relation_groups.items()):
        by_relation.append({"relation": relation, "subset": subset, **summarize(items)})

    return {
        "phase": 768,
        "title": "Semantic-Alias Phrase Closure",
        "model": model_name,
        "round": round_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "attn_implementation": attn_impl,
        "dtype": "bfloat16",
        "quantization": "off",
        "n_cases": len(rows),
        "by_subset": {name: summarize(items) for name, items in groups.items()},
        "by_relation_subset": by_relation,
        "strict_interpretation": "Phrase likelihood and short greedy generation test whether semantic alias closure survives beyond first-token ranking.",
    }


def run_model(args: argparse.Namespace) -> dict[str, Any]:
    relation_filter = set(args.relations.split(",")) if args.relations else {"category", "edible", "grows_on_tree"}
    cases = build_cases(args.max_cases, relation_filter)
    for case in cases:
        case["include_candidate_list"] = bool(args.include_candidate_list)
    phase767_rows = load_phase767_rows(args.model, args.phase767_round)
    missing = [case["case_id"] for case in cases if case["case_id"] not in phase767_rows]
    if missing:
        raise SystemExit(f"missing Phase 767 rows for {args.model}: {missing[:5]}")
    log(
        f"{args.model}/{args.round_name}: cases={len(cases)} relations={sorted(relation_filter)} "
        f"phase767_round={args.phase767_round} max_new_tokens={args.max_new_tokens}"
    )
    model, tokenizer, device, attn_impl = load_model_bf16(args.model, prefer_flash=not args.no_flash)
    try:
        rows = []
        for idx, case in enumerate(cases, 1):
            rows.append(audit_case(model, tokenizer, device, case, phase767_rows[case["case_id"]], args.max_new_tokens))
            if idx % args.log_every == 0 or idx == len(cases):
                log(f"{args.model}: phrase closure {idx}/{len(cases)} cases")
    finally:
        release_model(model)
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary = summarize_rows(args.model, args.round_name, rows, attn_impl)
    for root in (OUT_ROOT, RESULT_ROOT):
        out_dir = root / args.round_name
        write_jsonl(out_dir / f"phase768_{args.model}_rows.jsonl", rows)
        write_json(out_dir / f"phase768_{args.model}_summary.json", summary)
    print(json.dumps({"model": args.model, "round": args.round_name, "by_subset": summary["by_subset"]}, ensure_ascii=False, indent=2))
    return summary


def write_cross_summary(round_name: str) -> dict[str, Any]:
    summaries = []
    for model in MODELS:
        path = OUT_ROOT / round_name / f"phase768_{model}_summary.json"
        if path.exists():
            summaries.append(json.loads(path.read_text(encoding="utf-8")))
    payload = {
        "phase": 768,
        "title": "Semantic-Alias Phrase Closure",
        "round": round_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "status": "complete" if len(summaries) == len(MODELS) else "partial",
        "models": [s["model"] for s in summaries],
        "by_model": {s["model"]: s for s in summaries},
    }
    lines = [
        f"# Phase 768 Semantic-Alias Phrase Closure ({round_name})",
        "",
        f"- Status: `{payload['status']}`",
        "- Test: phrase likelihood over allowed values plus short greedy generation.",
        "- Input subset labels: Phase 767 semantic/exact closure rows.",
        "",
        "## By Subset",
        "",
        "| model | subset | n | semantic top1 | exact top1 | phrase top1 | generation match | phrase rank | phrase margin |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for summary in summaries:
        for subset, row in summary["by_subset"].items():
            lines.append(
                f"| {summary['model']} | `{subset}` | {row['n']} | {fmt(row['semantic_top1_rate'])} | "
                f"{fmt(row['exact_top1_rate'])} | {fmt(row['phrase_top1_rate'])} | "
                f"{fmt(row['generation_match_rate'])} | {fmt(row['mean_phrase_rank'])} | "
                f"{fmt(row['mean_phrase_margin'])} |"
            )
    lines += [
        "",
        "## Relation And Subset",
        "",
        "| model | relation | subset | n | phrase top1 | generation match | phrase rank | phrase margin |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for summary in summaries:
        for row in summary["by_relation_subset"]:
            lines.append(
                f"| {summary['model']} | `{row['relation']}` | `{row['subset']}` | {row['n']} | "
                f"{fmt(row['phrase_top1_rate'])} | {fmt(row['generation_match_rate'])} | "
                f"{fmt(row['mean_phrase_rank'])} | {fmt(row['mean_phrase_margin'])} |"
            )
    lines += [
        "",
        "## Strict Interpretation",
        "",
        "- Phrase top1 tests whether the best full allowed-value phrase is the target value.",
        "- Generation match is stricter: greedy continuation must start with the target semantic value.",
        "- If semantic-clean cases lose phrase top1, first-token semantic closure is not sufficient for phrase closure.",
        "- If semantic-only cases keep phrase top1, lexical capitalization is likely a surface realization issue.",
    ]
    markdown = "\n".join(lines) + "\n"
    for root in (OUT_ROOT, RESULT_ROOT):
        out_dir = root / round_name
        write_json(out_dir / "phase768_cross_model_summary.json", payload)
        (out_dir / "phase768_cross_model_summary.md").write_text(markdown, encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--round-name", default="smoke")
    parser.add_argument("--phase767-round", default="main")
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--relations", default="category,edible,grows_on_tree")
    parser.add_argument("--include-candidate-list", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-new-tokens", type=int, default=6)
    parser.add_argument("--log-every", type=int, default=18)
    parser.add_argument("--no-flash", action="store_true")
    parser.add_argument("--write-cross-summary", action="store_true")
    parser.add_argument("--hard-exit-after-model", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.dry_run:
        cases = build_cases(args.max_cases, set(args.relations.split(",")) if args.relations else None)
        print(json.dumps({"n_cases": len(cases), "sample_cases": cases[:6]}, ensure_ascii=False, indent=2))
        return
    if args.write_cross_summary:
        write_cross_summary(args.round_name)
        return
    if not args.model:
        raise SystemExit("--model is required unless --write-cross-summary or --dry-run")
    run_model(args)
    if args.hard_exit_after_model:
        raise SystemExit(0)


if __name__ == "__main__":
    main()
