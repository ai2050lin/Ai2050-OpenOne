#!/usr/bin/env python3
"""
Phase 590: Value Winner Selection Multi-Layer Patch
值候选胜出选择多层组合修补

Phase 589 showed single-layer single-component patches do not fix DS7B value winner selection.
Phase 590 tests cumulative patches across late layers and components.
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
from typing import Dict, List, Tuple

import torch

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from model_utils import get_layers, get_model_info, release_model  # noqa: E402
from phase584_gate_repair import CANDIDATE_VALUES, compute_full_string_logprob_batch, load_model_flash  # noqa: E402
from phase586_distributed_value_path_patch import build_cases, case_positions, random_same_norm, selected_layers  # noqa: E402
from phase587_value_winner_competition import winner_stats  # noqa: E402
from phase589_component_value_path_attribution import (  # noqa: E402
    collect_component_outputs,
    extract_tensor,
    get_component_module,
    replace_tensor,
)

OUT_ROOT = Path("results/glm5_phase590_value_winner_multilayer_patch")


def log(msg: str = "") -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def score_map(model, tokenizer, device, prompt: str, candidates: List[str]) -> Dict[str, float]:
    return {k: v[0] for k, v in compute_full_string_logprob_batch(model, tokenizer, device, prompt, candidates).items()}


def get_combos(layers: List[int]) -> Dict[str, List[Tuple[str, int]]]:
    mid, late = layers[-2], layers[-1]
    return {
        "residual_both": [("residual", mid), ("residual", late)],
        "attn_both": [("attn", mid), ("attn", late)],
        "mlp_both": [("mlp", mid), ("mlp", late)],
        "residual_attn_both": [("residual", mid), ("residual", late), ("attn", mid), ("attn", late)],
        "residual_mlp_both": [("residual", mid), ("residual", late), ("mlp", mid), ("mlp", late)],
        "all_both": [("residual", mid), ("residual", late), ("attn", mid), ("attn", late), ("mlp", mid), ("mlp", late)],
    }


def answer_ids(tokenizer, answer: str) -> List[int]:
    ids = tokenizer.encode(" " + answer, add_special_tokens=False)
    if not ids:
        ids = tokenizer.encode(answer, add_special_tokens=False)
    return ids


def patched_logprob_multi(model, tokenizer, device, prompt: str, answer: str,
                          patches: List[Tuple[str, int, int, torch.Tensor]], alpha: float) -> float:
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    ans_ids = answer_ids(tokenizer, answer)
    if not ans_ids:
        return -100.0
    all_ids = prompt_ids + ans_ids
    layers = get_layers(model)
    hooks = []
    for comp, li, pos, vec_cpu in patches:
        module = get_component_module(layers[li], comp)
        if module is None or pos < 0 or pos >= len(prompt_ids):
            continue
        vec = vec_cpu.to(device=device)

        def make_hook(patch_pos, patch_vec):
            def hook(_module, _inputs, output):
                h = extract_tensor(output)
                h_new = h.clone()
                h_new[0, patch_pos, :] = h_new[0, patch_pos, :] + alpha * patch_vec.to(dtype=h_new.dtype)
                return replace_tensor(output, h_new)
            return hook

        hooks.append(module.register_forward_hook(make_hook(pos, vec)))
    try:
        total = 0.0
        with torch.inference_mode():
            full_input = torch.tensor([all_ids], device=device)
            out = model(input_ids=full_input, return_dict=True)
            logits = out.logits[0].float()
            start = len(prompt_ids) - 1
            for i, tid in enumerate(ans_ids):
                pos = start + i
                if pos >= logits.shape[0]:
                    break
                total += float(torch.log_softmax(logits[pos], dim=-1)[tid].cpu())
        return total
    finally:
        for h in hooks:
            h.remove()


def patched_score_map_multi(model, tokenizer, device, prompt: str, candidates: List[str],
                            patches: List[Tuple[str, int, int, torch.Tensor]], alpha: float) -> Dict[str, float]:
    return {ans: patched_logprob_multi(model, tokenizer, device, prompt, ans, patches, alpha) for ans in candidates}


def build_patch_vectors(combo: List[Tuple[str, int]], pos_name: str, base_pos: Dict, donor_pos: Dict,
                        base_out: Dict, donor_out: Dict, mode: str, sample_seed: int):
    patches = []
    bp = base_pos.get(pos_name)
    dp = donor_pos.get(pos_name)
    if bp is None or dp is None:
        return patches
    for comp, li in combo:
        if li not in base_out.get(comp, {}) or li not in donor_out.get(comp, {}):
            continue
        if bp >= base_out[comp][li].shape[1] or dp >= donor_out[comp][li].shape[1]:
            continue
        delta = donor_out[comp][li][0, dp, :] - base_out[comp][li][0, bp, :]
        if mode == "random":
            delta = random_same_norm(delta, seed=sample_seed + li * 31 + len(comp))
        patches.append((comp, li, bp, delta))
    return patches


def run_model(args):
    model, tokenizer, device = load_model_flash(args.model)
    try:
        info = get_model_info(model, args.model)
        all_layers = selected_layers(info.n_layers)
        probe_layers = [all_layers[-2], all_layers[-1]]
        combos = get_combos(probe_layers)
        components = ["residual", "attn", "mlp"]
        cases = list(build_cases(args.n_tables, args.max_samples))
        values = CANDIDATE_VALUES[:4]
        log(f"{args.model}: n_layers={info.n_layers}, cases={len(cases)}, layers={probe_layers}")
        rows = []
        for si, case in enumerate(cases):
            correct = case["correct"]
            base = winner_stats(score_map(model, tokenizer, device, case["base_prompt"], values), correct)
            repair = winner_stats(score_map(model, tokenizer, device, case["repair_prompt"], values), correct)
            target_case = (not base["correct"]) and repair["correct"]
            base_pos = case_positions(tokenizer, case, case["base_prompt"], case["relation"])
            repair_pos = case_positions(tokenizer, case, case["repair_prompt"], case["repair_rel"])
            wrong_pos = case_positions(tokenizer, case, case["wrong_prompt"], case["wrong_rel"])
            base_out = collect_component_outputs(model, tokenizer, device, case["base_prompt"], probe_layers, components)
            repair_out = collect_component_outputs(model, tokenizer, device, case["repair_prompt"], probe_layers, components)
            wrong_out = collect_component_outputs(model, tokenizer, device, case["wrong_prompt"], probe_layers, components)
            for pos_name in ["prompt_last", "query_relation"]:
                for combo_name, combo in combos.items():
                    patch_sets = {
                        "repair_cumulative": build_patch_vectors(combo, pos_name, base_pos, repair_pos, base_out, repair_out, "repair", si * 101),
                        "wrong_relation_cumulative": build_patch_vectors(combo, pos_name, base_pos, wrong_pos, base_out, wrong_out, "repair", si * 101),
                        "random_cumulative": build_patch_vectors(combo, pos_name, base_pos, repair_pos, base_out, repair_out, "random", si * 101),
                    }
                    for mode, patches in patch_sets.items():
                        if not patches:
                            continue
                        patched = winner_stats(
                            patched_score_map_multi(model, tokenizer, device, case["base_prompt"], values, patches, args.alpha),
                            correct,
                        )
                        old_wrong = base["top_wrong"]
                        rows.append({
                            "sample_idx": si,
                            "case": {k: case[k] for k in ["tt_idx", "object", "relation", "category", "correct"]},
                            "target_case": target_case,
                            "position": pos_name,
                            "combo": combo_name,
                            "mode": mode,
                            "n_patches": len(patches),
                            "base": base,
                            "repair_prompt": repair,
                            "patch": patched,
                            "correct_gain": patched["correct_score"] - base["correct_score"],
                            "old_top_wrong_gain": patched["scores"].get(old_wrong, patched["top_wrong_score"]) - base["top_wrong_score"],
                            "margin_gain": patched["margin"] - base["margin"],
                        })
        summary = summarize(rows)
        return {
            "phase": 590,
            "model": args.model,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "n_layers": info.n_layers,
            "probe_layers": probe_layers,
            "n_cases": len(cases),
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
        key = f"{r['position']}|{r['combo']}|{r['mode']}"
        item = by_key.setdefault(key, {
            "position": r["position"],
            "combo": r["combo"],
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
    )[:12]
    for item in best:
        log(
            f"  {item['position']} {item['combo']} {item['mode']}: "
            f"switch={item['target_switch']}/{item['target_n']}, "
            f"cgain={item['mean_correct_gain']:.3f}, wgain={item['mean_top_wrong_gain']:.3f}, "
            f"mgain={item['mean_margin_gain']:.3f}"
        )
    return {"by_key": by_key, "best": best}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--n-tables", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=20)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--output-dir", default=str(OUT_ROOT))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--confirm", action="store_true")
    parser.add_argument("--hard-exit-after-model", action="store_true")
    args = parser.parse_args()

    if args.smoke:
        args.n_tables = 1
        args.max_samples = 3
        log("SMOKE TEST MODE")
    elif args.confirm:
        args.n_tables = max(args.n_tables, 5)
        args.max_samples = max(args.max_samples, 24)
        log("CONFIRMATION TEST MODE")

    t0 = time.time()
    result = run_model(args)
    result["total_time_min"] = round((time.time() - t0) / 60.0, 2)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_smoke" if args.smoke else ("_confirm" if args.confirm else "")
    out_path = out_dir / f"phase590_{args.model}_value_winner_multilayer_patch{suffix}.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    log(f"Wrote {out_path}")
    log(f"Total time: {result['total_time_min']} min")
    if args.hard_exit_after_model:
        os._exit(0)


if __name__ == "__main__":
    main()
