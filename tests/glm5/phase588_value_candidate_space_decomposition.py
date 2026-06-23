#!/usr/bin/env python3
"""
Phase 588: Value Candidate Space Decomposition
值候选空间分解

Phase 587 found that repair delta raises correct and top-wrong together.
This phase tests simple decompositions of repair delta:
  - full repair delta
  - remove common value-space component
  - common-only component
  - correct-vs-wrong contrast component
  - top-wrong suppression / correct boost
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
sys.path.insert(0, str(Path(__file__).resolve().parent))

from model_utils import get_model_info, release_model  # noqa: E402
from phase584_gate_repair import CANDIDATE_VALUES, load_model_flash, compute_full_string_logprob_batch  # noqa: E402
from phase586_distributed_value_path_patch import (  # noqa: E402
    build_cases,
    case_positions,
    get_hidden,
    patch_full_logprob,
    random_same_norm,
    selected_layers,
)
from phase587_value_winner_competition import winner_stats, answer_first_token  # noqa: E402

OUT_ROOT = Path("results/glm5_phase588_value_candidate_space_decomposition")


def log(msg: str = "") -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def score_map(model, tokenizer, device, prompt: str, candidates: List[str]) -> Dict[str, float]:
    return {k: v[0] for k, v in compute_full_string_logprob_batch(model, tokenizer, device, prompt, candidates).items()}


def patched_score_map(model, tokenizer, device, prompt: str, candidates: List[str],
                      layer_idx: int, patch_pos: int, vec: torch.Tensor, alpha: float) -> Dict[str, float]:
    return {
        ans: patch_full_logprob(model, tokenizer, device, prompt, ans, layer_idx, patch_pos, vec, "add", alpha)[0]
        for ans in candidates
    }


def unembed_vec(model, tokenizer, value: str) -> torch.Tensor:
    weight = model.get_output_embeddings().weight.detach().float().cpu()
    return weight[answer_first_token(tokenizer, value)]


def project(vec: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
    d = direction.float()
    denom = torch.dot(d, d).clamp_min(1e-8)
    return torch.dot(vec.float(), d) / denom * d


def norm_match(vec: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    v = vec.float()
    v_norm = torch.linalg.vector_norm(v).clamp_min(1e-8)
    ref_norm = torch.linalg.vector_norm(reference.float()).clamp_min(1e-8)
    return v / v_norm * ref_norm


def build_components(model, tokenizer, values: List[str], correct: str, top_wrong: str, delta: torch.Tensor) -> Dict[str, torch.Tensor]:
    value_vecs = [unembed_vec(model, tokenizer, v) for v in values]
    common = torch.stack(value_vecs).mean(dim=0)
    correct_vec = unembed_vec(model, tokenizer, correct)
    wrong_vec = unembed_vec(model, tokenizer, top_wrong)
    contrast = correct_vec - wrong_vec
    suppress_wrong = -wrong_vec
    boost_correct = correct_vec

    common_part = project(delta, common)
    contrast_part = project(delta, contrast)
    return {
        "full_delta": delta,
        "common_only": common_part,
        "remove_common": delta - common_part,
        "contrast_only": norm_match(contrast_part, delta),
        "remove_contrast": delta - contrast_part,
        "boost_correct": norm_match(boost_correct, delta),
        "suppress_top_wrong": norm_match(suppress_wrong, delta),
        "boost_minus_suppress": norm_match(correct_vec - wrong_vec, delta),
    }


def run_model(args):
    model, tokenizer, device = load_model_flash(args.model)
    try:
        info = get_model_info(model, args.model)
        layers = selected_layers(info.n_layers)
        probe_layers = [layers[-2], layers[-1]] if len(layers) >= 2 else [layers[-1]]
        cases = list(build_cases(args.n_tables, args.max_samples))
        values = CANDIDATE_VALUES[:4]
        log(f"{args.model}: n_layers={info.n_layers}, cases={len(cases)}, layers={probe_layers}")

        rows = []
        for si, case in enumerate(cases):
            correct = case["correct"]
            base = winner_stats(score_map(model, tokenizer, device, case["base_prompt"], values), correct)
            repair = winner_stats(score_map(model, tokenizer, device, case["repair_prompt"], values), correct)
            target_case = (not base["correct"]) and repair["correct"]
            base_h = get_hidden(model, tokenizer, device, case["base_prompt"], probe_layers)
            repair_h = get_hidden(model, tokenizer, device, case["repair_prompt"], probe_layers)
            base_pos = case_positions(tokenizer, case, case["base_prompt"], case["relation"])
            repair_pos = case_positions(tokenizer, case, case["repair_prompt"], case["repair_rel"])

            for pos_name in ["prompt_last", "query_relation"]:
                bp = base_pos.get(pos_name)
                rp = repair_pos.get(pos_name)
                if bp is None or rp is None:
                    continue
                for li in probe_layers:
                    if bp >= base_h[li].shape[0] or rp >= repair_h[li].shape[0]:
                        continue
                    delta = repair_h[li][rp] - base_h[li][bp]
                    components = build_components(model, tokenizer, values, correct, base["top_wrong"], delta)
                    components["random_same_norm"] = random_same_norm(delta, seed=si * 997 + li)
                    for mode, vec in components.items():
                        patched = winner_stats(
                            patched_score_map(model, tokenizer, device, case["base_prompt"], values, li, bp, vec, args.alpha),
                            correct,
                        )
                        old_wrong = base["top_wrong"]
                        rows.append({
                            "sample_idx": si,
                            "case": {k: case[k] for k in ["tt_idx", "object", "relation", "category", "correct"]},
                            "target_case": target_case,
                            "position": pos_name,
                            "layer": li,
                            "mode": mode,
                            "base": base,
                            "repair_prompt": repair,
                            "patch": patched,
                            "correct_gain": patched["correct_score"] - base["correct_score"],
                            "old_top_wrong_gain": patched["scores"].get(old_wrong, patched["top_wrong_score"]) - base["top_wrong_score"],
                            "margin_gain": patched["margin"] - base["margin"],
                        })
        summary = summarize(rows)
        return {
            "phase": 588,
            "model": args.model,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "n_layers": info.n_layers,
            "probe_layers": probe_layers,
            "n_cases": len(cases),
            "n_target_cases": sum(1 for r in rows if r["target_case"] and r["mode"] == "full_delta" and r["position"] == "prompt_last" and r["layer"] == probe_layers[-1]),
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
    by_key = {}
    for r in rows:
        key = f"{r['position']}|L{r['layer']}|{r['mode']}"
        item = by_key.setdefault(key, {
            "position": r["position"],
            "layer": r["layer"],
            "mode": r["mode"],
            "n": 0,
            "target_n": 0,
            "target_switch": 0,
            "mean_correct_gain": 0.0,
            "mean_top_wrong_gain": 0.0,
            "mean_margin_gain": 0.0,
            "correct_up_competitor_up": 0,
            "correct_up_margin_negative": 0,
        })
        item["n"] += 1
        if r["target_case"]:
            item["target_n"] += 1
            item["target_switch"] += int(r["patch"]["correct"])
            item["mean_correct_gain"] += r["correct_gain"]
            item["mean_top_wrong_gain"] += r["old_top_wrong_gain"]
            item["mean_margin_gain"] += r["margin_gain"]
            if r["correct_gain"] > 0 and r["old_top_wrong_gain"] > 0:
                item["correct_up_competitor_up"] += 1
            if r["correct_gain"] > 0 and r["patch"]["margin"] < 0:
                item["correct_up_margin_negative"] += 1
    for item in by_key.values():
        tn = max(1, item["target_n"])
        item["target_switch_rate"] = item["target_switch"] / tn
        item["mean_correct_gain"] /= tn
        item["mean_top_wrong_gain"] /= tn
        item["mean_margin_gain"] /= tn
    best = sorted(
        by_key.values(),
        key=lambda x: (x["target_n"], x["target_switch_rate"], x["mean_margin_gain"], x["mean_correct_gain"]),
        reverse=True,
    )[:16]
    for item in best:
        log(
            f"  {item['position']} L{item['layer']} {item['mode']}: "
            f"switch={item['target_switch']}/{item['target_n']}, "
            f"cgain={item['mean_correct_gain']:.3f}, wgain={item['mean_top_wrong_gain']:.3f}, "
            f"mgain={item['mean_margin_gain']:.3f}"
        )
    return {"by_key": by_key, "best": best}


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
    out_path = out_dir / f"phase588_{args.model}_value_candidate_space_decomposition{suffix}.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    log(f"Wrote {out_path}")
    log(f"Total time: {result['total_time_min']} min")

    if args.hard_exit_after_model:
        os._exit(0)


if __name__ == "__main__":
    main()
