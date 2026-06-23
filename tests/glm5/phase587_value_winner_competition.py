#!/usr/bin/env python3
"""
Phase 587: Value Winner Competition Audit
值候选胜出竞争审计

Phase 586 showed that DS7B can gain large correct-value logprob without winner switch.
Phase 587 records all candidate value scores and audits support vs competitor suppression.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import torch

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from model_utils import get_model_info, release_model  # noqa: E402
from phase584_gate_repair import (  # noqa: E402
    CANDIDATE_VALUES,
    compute_full_string_logprob_batch,
    load_model_flash,
)
from phase586_distributed_value_path_patch import (  # noqa: E402
    build_cases,
    case_positions,
    get_hidden,
    patch_full_logprob,
    random_same_norm,
    selected_layers,
)

OUT_ROOT = Path("results/glm5_phase587_value_winner_competition")


def log(msg: str = "") -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def score_map(model, tokenizer, device, prompt: str, candidates: List[str]) -> Dict[str, float]:
    return {k: v[0] for k, v in compute_full_string_logprob_batch(model, tokenizer, device, prompt, candidates).items()}


def winner_stats(scores: Dict[str, float], correct: str) -> Dict:
    pred = max(scores, key=scores.get)
    competitors = {k: v for k, v in scores.items() if k != correct}
    top_wrong = max(competitors, key=competitors.get)
    margin = scores[correct] - competitors[top_wrong]
    return {
        "pred": pred,
        "correct": pred == correct,
        "correct_score": scores[correct],
        "top_wrong": top_wrong,
        "top_wrong_score": competitors[top_wrong],
        "margin": margin,
        "scores": scores,
    }


def patched_score_map(model, tokenizer, device, prompt: str, candidates: List[str],
                      layer_idx: int, patch_pos: int, patch_vec: torch.Tensor,
                      mode: str, alpha: float) -> Dict[str, float]:
    return {
        ans: patch_full_logprob(
            model, tokenizer, device, prompt, ans, layer_idx, patch_pos, patch_vec, mode, alpha
        )[0]
        for ans in candidates
    }


def answer_first_token(tokenizer, answer: str) -> int:
    ids = tokenizer.encode(" " + answer, add_special_tokens=False)
    if not ids:
        ids = tokenizer.encode(answer, add_special_tokens=False)
    return ids[0]


def get_unembedding_direction(model, tokenizer, correct: str, wrong: str) -> torch.Tensor:
    emb = model.get_output_embeddings().weight.detach().float().cpu()
    c_id = answer_first_token(tokenizer, correct)
    w_id = answer_first_token(tokenizer, wrong)
    return emb[c_id] - emb[w_id]


def scaled_direction(direction: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    d = direction.float()
    d_norm = torch.linalg.vector_norm(d).clamp_min(1e-8)
    ref_norm = torch.linalg.vector_norm(reference.float()).clamp_min(1e-8)
    return d / d_norm * ref_norm


def run_model(args):
    model, tokenizer, device = load_model_flash(args.model)
    try:
        info = get_model_info(model, args.model)
        layers = selected_layers(info.n_layers)
        late_layer = layers[-1]
        mid_layer = layers[-2] if len(layers) >= 2 else layers[-1]
        cases = list(build_cases(args.n_tables, args.max_samples))
        log(f"{args.model}: n_layers={info.n_layers}, cases={len(cases)}, late_layer=L{late_layer}")

        rows = []
        target_rows = []
        for si, case in enumerate(cases):
            candidates = CANDIDATE_VALUES[:4]
            correct = case["correct"]
            base_scores = score_map(model, tokenizer, device, case["base_prompt"], candidates)
            repair_scores = score_map(model, tokenizer, device, case["repair_prompt"], candidates)
            base = winner_stats(base_scores, correct)
            repair = winner_stats(repair_scores, correct)
            target_case = (not base["correct"]) and repair["correct"]

            base_h = get_hidden(model, tokenizer, device, case["base_prompt"], [mid_layer, late_layer])
            repair_h = get_hidden(model, tokenizer, device, case["repair_prompt"], [mid_layer, late_layer])
            wrong_h = get_hidden(model, tokenizer, device, case["wrong_prompt"], [mid_layer, late_layer])
            base_pos = case_positions(tokenizer, case, case["base_prompt"], case["relation"])
            repair_pos = case_positions(tokenizer, case, case["repair_prompt"], case["repair_rel"])
            wrong_pos = case_positions(tokenizer, case, case["wrong_prompt"], case["wrong_rel"])
            patch_specs = []
            for pos_name in ["prompt_last", "query_relation"]:
                bp = base_pos.get(pos_name)
                rp = repair_pos.get(pos_name)
                wp = wrong_pos.get(pos_name)
                if bp is None or rp is None or wp is None:
                    continue
                for li in [mid_layer, late_layer]:
                    if bp >= base_h[li].shape[0] or rp >= repair_h[li].shape[0] or wp >= wrong_h[li].shape[0]:
                        continue
                    delta = repair_h[li][rp] - base_h[li][bp]
                    wrong_delta = wrong_h[li][wp] - base_h[li][bp]
                    random_delta = random_same_norm(delta, seed=si * 101 + li)
                    margin_dir = scaled_direction(
                        get_unembedding_direction(model, tokenizer, correct, base["top_wrong"]),
                        delta,
                    )
                    patch_specs += [
                        (f"{pos_name}|L{li}|add_repair", li, bp, delta, "add"),
                        (f"{pos_name}|L{li}|replace_repair", li, bp, repair_h[li][rp], "replace"),
                        (f"{pos_name}|L{li}|wrong_relation", li, bp, wrong_delta, "add"),
                        (f"{pos_name}|L{li}|random", li, bp, random_delta, "add"),
                        (f"{pos_name}|L{li}|readout_margin", li, bp, margin_dir, "add"),
                    ]

            row = {
                "sample_idx": si,
                "case": {k: case[k] for k in ["tt_idx", "object", "relation", "category", "correct"]},
                "target_case": target_case,
                "base": base,
                "repair_prompt": repair,
                "patches": {},
            }
            for name, li, bp, vec, mode in patch_specs:
                patched = winner_stats(
                    patched_score_map(
                        model, tokenizer, device, case["base_prompt"], candidates,
                        li, bp, vec, mode, args.alpha
                    ),
                    correct,
                )
                row["patches"][name] = patched
            rows.append(row)
            if target_case:
                target_rows.append(row)

        summary = summarize(rows)
        return {
            "phase": 587,
            "model": args.model,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "n_layers": info.n_layers,
            "probe_layers": layers,
            "n_cases": len(cases),
            "n_target_cases": len(target_rows),
            "alpha": args.alpha,
            "summary": summary,
            "rows": rows,
        }
    finally:
        release_model(model)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def summarize(rows: List[Dict]) -> Dict:
    base_n = len(rows)
    target_rows = [r for r in rows if r["target_case"]]
    patch_keys = sorted({k for r in rows for k in r["patches"]})
    by_patch = {}
    for key in patch_keys:
        all_items = [r["patches"][key] for r in rows if key in r["patches"]]
        tgt_pairs = [(r, r["patches"][key]) for r in target_rows if key in r["patches"]]
        item = {
            "n": len(all_items),
            "target_n": len(tgt_pairs),
            "patch_accuracy": sum(1 for p in all_items if p["correct"]) / max(1, len(all_items)),
            "target_switch": sum(1 for _, p in tgt_pairs if p["correct"]),
            "mean_correct_gain_target": 0.0,
            "mean_top_wrong_gain_target": 0.0,
            "mean_margin_gain_target": 0.0,
            "correct_up_competitor_up": 0,
            "correct_up_margin_negative": 0,
        }
        for r, p in tgt_pairs:
            base = r["base"]
            old_wrong = base["top_wrong"]
            correct_gain = p["correct_score"] - base["correct_score"]
            wrong_gain = p["scores"].get(old_wrong, p["top_wrong_score"]) - base["top_wrong_score"]
            margin_gain = p["margin"] - base["margin"]
            item["mean_correct_gain_target"] += correct_gain
            item["mean_top_wrong_gain_target"] += wrong_gain
            item["mean_margin_gain_target"] += margin_gain
            if correct_gain > 0 and wrong_gain > 0:
                item["correct_up_competitor_up"] += 1
            if correct_gain > 0 and p["margin"] < 0:
                item["correct_up_margin_negative"] += 1
        tn = max(1, item["target_n"])
        item["target_switch_rate"] = item["target_switch"] / tn
        item["mean_correct_gain_target"] /= tn
        item["mean_top_wrong_gain_target"] /= tn
        item["mean_margin_gain_target"] /= tn
        by_patch[key] = item

    best = sorted(
        by_patch.items(),
        key=lambda kv: (
            kv[1]["target_n"],
            kv[1]["target_switch_rate"],
            kv[1]["mean_margin_gain_target"],
            kv[1]["mean_correct_gain_target"],
        ),
        reverse=True,
    )[:12]
    for key, item in best:
        log(
            f"  {key}: target={item['target_switch']}/{item['target_n']}, "
            f"correct_gain={item['mean_correct_gain_target']:.3f}, "
            f"wrong_gain={item['mean_top_wrong_gain_target']:.3f}, "
            f"margin_gain={item['mean_margin_gain_target']:.3f}"
        )
    return {
        "n_cases": base_n,
        "n_target_cases": len(target_rows),
        "by_patch": by_patch,
        "best": [{"patch": k, **v} for k, v in best],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--n-tables", type=int, default=5)
    parser.add_argument("--max-samples", type=int, default=24)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--output-dir", default=str(OUT_ROOT))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--confirm", action="store_true")
    parser.add_argument("--hard-exit-after-model", action="store_true")
    args = parser.parse_args()

    if args.smoke:
        args.n_tables = 1
        args.max_samples = 4
        log("SMOKE TEST MODE")
    elif args.confirm:
        args.n_tables = max(args.n_tables, 6)
        args.max_samples = max(args.max_samples, 32)
        log("CONFIRMATION TEST MODE")

    t0 = time.time()
    result = run_model(args)
    result["total_time_min"] = round((time.time() - t0) / 60.0, 2)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_smoke" if args.smoke else ("_confirm" if args.confirm else "")
    out_path = out_dir / f"phase587_{args.model}_value_winner_competition{suffix}.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    log(f"Wrote {out_path}")
    log(f"Total time: {result['total_time_min']} min")

    if args.hard_exit_after_model:
        os._exit(0)


if __name__ == "__main__":
    main()
