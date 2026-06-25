#!/usr/bin/env python3
"""
Phase 637: Newline Prior Suppression Source Audit.

Phase 636 localized DS7B token0 failure to newline / format-continuation prior.
This phase tests prompt ablations that may suppress the newline prior, while
tracking target and non-target side effects separately.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import torch

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))
sys.path.insert(0, str(ROOT / "tests" / "gpt5"))

from model_utils import get_model_info, release_model  # noqa: E402
from phase584_gate_repair import CANDIDATE_VALUES, load_model_flash  # noqa: E402
from phase587_value_winner_competition import winner_stats  # noqa: E402
from phase597_state_conditioned_mlp_generation_audit import score_map  # noqa: E402
from phase609_query_oproj_head_decomposition import answer_ids  # noqa: E402
from phase612_source_aligned_pattern_content_split import build_aligned_cases  # noqa: E402
from phase628_prefix_format_semantic_integration import generation_eval, token_strings  # noqa: E402
from phase636_prefix_competitor_ladder_audit import clean_token, ladder_for_logits  # noqa: E402


OUT_ROOT = Path("results/glm5_phase637_newline_prior_suppression_source_audit")
PROMPT_VARIANTS = [
    "original",
    "no_qmark",
    "period",
    "inline_answer",
    "short_only",
    "no_explain",
    "no_qmark_short",
    "value_label",
    "direct_value_label",
]
SUBJECT_KINDS = ["base_subject", "repair_subject"]


def log(msg: str = "") -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def prompt_common(prompt: str) -> str:
    marker = "Question: "
    if marker not in prompt:
        return prompt
    return prompt.rsplit(marker, 1)[0]


def make_prompt(case: Dict, subject_kind: str, variant: str) -> str:
    subject = case["object"] if subject_kind == "base_subject" else case["category"]
    rel = case["relation"]
    common = prompt_common(case["base_prompt"])
    question = f"{subject} {rel}"
    if variant == "original":
        return common + f"Question: {question} ?\nAnswer:"
    if variant == "no_qmark":
        return common + f"Question: {question}\nAnswer:"
    if variant == "period":
        return common + f"Question: {question}.\nAnswer:"
    if variant == "inline_answer":
        return common + f"Question: {question} ? Answer:"
    if variant == "short_only":
        return common + f"Instruction: Answer with only the value.\nQuestion: {question} ?\nAnswer:"
    if variant == "no_explain":
        return common + f"Instruction: Do not explain. Answer with only the value.\nQuestion: {question} ?\nAnswer:"
    if variant == "no_qmark_short":
        return common + f"Instruction: Answer with only the value.\nQuestion: {question}\nAnswer:"
    if variant == "value_label":
        return common + f"Question: {question} ?\nValue:"
    if variant == "direct_value_label":
        return common + f"Instruction: Return only the value.\nQuestion: {question} ?\nValue:"
    raise ValueError(variant)


def greedy_generate(model, tokenizer, device, prompt: str, max_new_tokens: int) -> Dict:
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    gen = []
    top5 = []
    with torch.inference_mode():
        for _step in range(max_new_tokens):
            logits = model(input_ids=torch.tensor([ids], device=device), return_dict=True).logits[0, -1].float()
            topv, topi = torch.topk(torch.log_softmax(logits, dim=-1), k=5)
            top5.append([
                {"id": int(i), "text": tokenizer.decode([int(i)]), "logprob": float(v)}
                for v, i in zip(topv.cpu(), topi.cpu())
            ])
            tid = int(torch.argmax(logits).item())
            gen.append(tid)
            ids.append(tid)
    return {"ids": gen, "tokens": token_strings(tokenizer, gen), "text": tokenizer.decode(gen), "top5": top5}


def token0_logits(model, tokenizer, device, prompt: str) -> torch.Tensor:
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    with torch.inference_mode():
        return model(input_ids=torch.tensor([ids], device=device), return_dict=True).logits[0, -1].detach().float().cpu()


def summarize(rows: List[Dict]) -> Dict:
    by_mode_split = {}
    for row in rows:
        key = (row["mode"], row["split"])
        item = by_mode_split.setdefault(key, {
            "mode": row["mode"],
            "subject_kind": row["subject_kind"],
            "variant": row["variant"],
            "split": row["split"],
            "n": 0,
            "tok0_hit": 0,
            "exact": 0,
            "wrong_exact": 0,
            "newline_top0": 0,
            "explanation_top0": 0,
            "space_top0": 0,
            "word_top0": 0,
            "sum_rank": 0.0,
            "sum_prefix_minus_newline": 0.0,
            "sum_prefix_minus_top": 0.0,
            "top0_text": {},
            "top0_category": {},
        })
        item["n"] += 1
        item["tok0_hit"] += int(row["top0_id"] == row["prefix_id"])
        item["exact"] += int(row["eval"]["exact_correct"])
        item["wrong_exact"] += int(row["eval"]["exact_wrong"])
        item["newline_top0"] += int(row["top0_category"] == "newline")
        item["explanation_top0"] += int(row["top0_category"] == "explanation")
        item["space_top0"] += int(row["top0_category"] == "space")
        item["word_top0"] += int(row["top0_category"] == "word")
        item["sum_rank"] += row["prefix_rank"]
        item["sum_prefix_minus_top"] += row["prefix_margin_vs_top"]
        item["sum_prefix_minus_newline"] += row["prefix_minus_newline"]
        item["top0_text"].setdefault(row["top0_text_clean"], 0)
        item["top0_text"][row["top0_text_clean"]] += 1
        item["top0_category"].setdefault(row["top0_category"], 0)
        item["top0_category"][row["top0_category"]] += 1
    out = []
    for item in by_mode_split.values():
        n = max(1, item["n"])
        row = dict(item)
        row["tok0_rate"] = item["tok0_hit"] / n
        row["exact_rate"] = item["exact"] / n
        row["wrong_exact_rate"] = item["wrong_exact"] / n
        row["newline_top0_rate"] = item["newline_top0"] / n
        row["mean_prefix_rank"] = item["sum_rank"] / n
        row["mean_prefix_minus_newline"] = item["sum_prefix_minus_newline"] / n
        row["mean_prefix_margin_vs_top"] = item["sum_prefix_minus_top"] / n
        row["top0_text"] = dict(sorted(row["top0_text"].items(), key=lambda kv: kv[1], reverse=True)[:8])
        row["top0_category"] = dict(sorted(row["top0_category"].items(), key=lambda kv: kv[1], reverse=True))
        out.append(row)
    out.sort(key=lambda x: (
        x["split"],
        x["subject_kind"],
        x["newline_top0_rate"],
        -x["tok0_rate"],
        x["mean_prefix_rank"],
    ))
    return {"by_mode_split": out}


def run_model(args) -> Dict:
    model, tokenizer, device = load_model_flash(args.model)
    try:
        info = get_model_info(model, args.model)
        values = CANDIDATE_VALUES[:4]
        raw_cases = list(build_aligned_cases(args.n_tables, args.max_samples))
        value_prefix_ids = {answer_ids(tokenizer, v)[0] for v in values}
        max_new_tokens = max(len(answer_ids(tokenizer, v)) for v in values)
        rows = []
        examples = []
        target_count = 0
        log(f"{args.model}: variants={PROMPT_VARIANTS}, raw_cases={len(raw_cases)}")

        for si, case in enumerate(raw_cases):
            base_scores = score_map(model, tokenizer, device, case["base_prompt"], values)
            repair_scores = score_map(model, tokenizer, device, case["repair_prompt"], values)
            base = winner_stats(base_scores, case["correct"])
            repair = winner_stats(repair_scores, case["correct"])
            target_case = (not base["correct"]) and repair["correct"]
            target_count += int(target_case)
            split = "target" if target_case else "non_target"
            if args.target_only and not target_case:
                continue

            correct_ids = answer_ids(tokenizer, case["correct"])
            old_wrong_ids = answer_ids(tokenizer, base["top_wrong"])
            prefix_id = correct_ids[0]
            old_wrong_prefix_id = old_wrong_ids[0]

            for subject_kind in SUBJECT_KINDS:
                for variant in PROMPT_VARIANTS:
                    prompt = make_prompt(case, subject_kind, variant)
                    logits = token0_logits(model, tokenizer, device, prompt)
                    ladder = ladder_for_logits(
                        tokenizer,
                        logits,
                        prefix_id,
                        old_wrong_prefix_id,
                        value_prefix_ids,
                        args.top_k,
                    )
                    newline_group = ladder["groups"].get("newline")
                    prefix_minus_newline = newline_group["prefix_minus_group_max"] if newline_group else 99.0
                    gen = greedy_generate(model, tokenizer, device, prompt, max_new_tokens)
                    ev = generation_eval(gen, correct_ids, old_wrong_ids)
                    mode = f"{subject_kind}__{variant}"
                    row = {
                        "sample_idx": si,
                        "split": split,
                        "mode": mode,
                        "subject_kind": subject_kind,
                        "variant": variant,
                        "case": {k: case[k] for k in ["tt_idx", "object", "relation", "category", "correct"]},
                        "old_top_wrong": base["top_wrong"],
                        "prefix_id": prefix_id,
                        "prefix_text": tokenizer.decode([prefix_id]),
                        "prefix_rank": ladder["prefix_rank"],
                        "top0_id": ladder["top0_id"],
                        "top0_text": ladder["top0_text"],
                        "top0_text_clean": clean_token(ladder["top0_text"]),
                        "top0_category": ladder["top0_category"],
                        "prefix_margin_vs_top": ladder["prefix_logit"] - logits[ladder["top0_id"]].item(),
                        "prefix_minus_newline": prefix_minus_newline,
                        "top": ladder["top"][:8],
                        "groups": ladder["groups"],
                        "generation_text": gen["text"],
                        "eval": ev,
                    }
                    rows.append(row)
                    if len(examples) < args.example_limit:
                        examples.append(row)

        summary = summarize(rows)
        log("Best target prompt variants:")
        target_rows = [r for r in summary["by_mode_split"] if r["split"] == "target" and r["subject_kind"] == "repair_subject"]
        target_rows = sorted(target_rows, key=lambda x: (x["newline_top0_rate"], -x["tok0_rate"], -x["exact_rate"], x["mean_prefix_rank"]))
        for item in target_rows[:16]:
            log(
                f"  {item['mode']}: n={item['n']} tok0={item['tok0_hit']}/{item['n']} "
                f"exact={item['exact']}/{item['n']} newline={item['newline_top0']}/{item['n']} "
                f"rank={item['mean_prefix_rank']:.1f} p-newline={item['mean_prefix_minus_newline']:.3f}"
            )
        return {
            "phase": 637,
            "model": args.model,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "n_layers": info.n_layers,
            "variants": PROMPT_VARIANTS,
            "subject_kinds": SUBJECT_KINDS,
            "top_k": args.top_k,
            "n_raw_cases": len(raw_cases),
            "n_target_cases_seen": target_count,
            "n_rows": len(rows),
            "target_only": args.target_only,
            "summary": summary,
            "examples": examples,
            "rows": rows if args.save_rows else examples,
        }
    finally:
        release_model(model)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--n-tables", type=int, default=8)
    parser.add_argument("--max-samples", type=int, default=96)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--target-only", action="store_true", default=False)
    parser.add_argument("--save-rows", action="store_true")
    parser.add_argument("--example-limit", type=int, default=120)
    parser.add_argument("--output-dir", default=str(OUT_ROOT))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--confirm", action="store_true")
    parser.add_argument("--hard-exit-after-model", action="store_true")
    args = parser.parse_args()

    if args.smoke:
        args.n_tables = 1
        args.max_samples = 4
        args.top_k = min(args.top_k, 12)
        args.example_limit = 24
        log("SMOKE TEST MODE")
    if args.confirm:
        args.n_tables = max(args.n_tables, 32)
        args.max_samples = max(args.max_samples, 256)
        args.top_k = max(args.top_k, 20)
        args.example_limit = max(args.example_limit, 180)
        log("CONFIRMATION TEST MODE")

    t0 = time.time()
    result = run_model(args)
    result["total_time_min"] = (time.time() - t0) / 60.0
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "smoke" if args.smoke else "confirm" if args.confirm else "run"
    out_path = out_dir / f"phase637_{args.model}_newline_prior_suppression_source_audit_{suffix}.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"Wrote {out_path}")
    log(f"Total time: {result['total_time_min']:.2f} min")
    if args.hard_exit_after_model:
        os._exit(0)


if __name__ == "__main__":
    main()
