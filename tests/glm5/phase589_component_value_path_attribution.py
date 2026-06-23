#!/usr/bin/env python3
"""
Phase 589: Component-Level Value Path Attribution
组件级值路径归因

Phase 588 showed value co-activation is not explained by simple lm_head geometry.
This phase patches component outputs at late layers:
  - residual/layer output
  - attention output
  - MLP output
and measures correct/top-wrong/margin.
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
from typing import Dict, List, Optional

import torch

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from model_utils import get_layers, get_model_info, release_model  # noqa: E402
from phase584_gate_repair import CANDIDATE_VALUES, compute_full_string_logprob_batch, load_model_flash  # noqa: E402
from phase586_distributed_value_path_patch import (  # noqa: E402
    build_cases,
    case_positions,
    random_same_norm,
    selected_layers,
)
from phase587_value_winner_competition import winner_stats  # noqa: E402

OUT_ROOT = Path("results/glm5_phase589_component_value_path_attribution")


def log(msg: str = "") -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def score_map(model, tokenizer, device, prompt: str, candidates: List[str]) -> Dict[str, float]:
    return {k: v[0] for k, v in compute_full_string_logprob_batch(model, tokenizer, device, prompt, candidates).items()}


def get_component_module(layer, component: str):
    if component == "residual":
        return layer
    if component == "attn":
        for name in ["self_attn", "attention", "attn"]:
            if hasattr(layer, name):
                return getattr(layer, name)
    if component == "mlp":
        if hasattr(layer, "mlp"):
            return layer.mlp
    return None


def extract_tensor(output):
    if isinstance(output, tuple):
        return output[0]
    return output


def replace_tensor(output, tensor):
    if isinstance(output, tuple):
        return (tensor,) + output[1:]
    return tensor


def collect_component_outputs(model, tokenizer, device, prompt: str, layer_indices: List[int],
                              components: List[str]) -> Dict[str, Dict[int, torch.Tensor]]:
    layers = get_layers(model)
    captured: Dict[str, Dict[int, torch.Tensor]] = {c: {} for c in components}
    hooks = []
    for comp in components:
        for li in layer_indices:
            module = get_component_module(layers[li], comp)
            if module is None:
                continue

            def make_hook(component_name, layer_idx):
                def hook(_module, _inputs, output):
                    captured[component_name][layer_idx] = extract_tensor(output).detach().float().cpu()
                return hook

            hooks.append(module.register_forward_hook(make_hook(comp, li)))
    try:
        input_ids = torch.tensor([tokenizer.encode(prompt, add_special_tokens=False)], device=device)
        with torch.inference_mode():
            model(input_ids=input_ids, return_dict=True)
    finally:
        for h in hooks:
            h.remove()
    return captured


def patch_full_logprob_component(model, tokenizer, device, prompt: str, answer: str,
                                 layer_idx: int, component: str, patch_pos: int,
                                 patch_vec: torch.Tensor, alpha: float) -> float:
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    answer_ids = tokenizer.encode(" " + answer, add_special_tokens=False)
    if not answer_ids:
        answer_ids = tokenizer.encode(answer, add_special_tokens=False)
    if not answer_ids or patch_pos < 0 or patch_pos >= len(prompt_ids):
        return -100.0
    all_ids = prompt_ids + answer_ids
    layers = get_layers(model)
    module = get_component_module(layers[layer_idx], component)
    if module is None:
        return -100.0
    vec = patch_vec.to(device=device)

    def hook(_module, _inputs, output):
        h = extract_tensor(output)
        h_new = h.clone()
        h_new[0, patch_pos, :] = h_new[0, patch_pos, :] + alpha * vec.to(dtype=h_new.dtype)
        return replace_tensor(output, h_new)

    handle = module.register_forward_hook(hook)
    try:
        total = 0.0
        with torch.inference_mode():
            full_input = torch.tensor([all_ids], device=device)
            out = model(input_ids=full_input, return_dict=True)
            logits = out.logits[0].float()
            start = len(prompt_ids) - 1
            for i, tid in enumerate(answer_ids):
                pos = start + i
                if pos >= logits.shape[0]:
                    break
                total += float(torch.log_softmax(logits[pos], dim=-1)[tid].cpu())
        return total
    finally:
        handle.remove()


def patched_score_map(model, tokenizer, device, prompt: str, candidates: List[str],
                      layer_idx: int, component: str, patch_pos: int,
                      patch_vec: torch.Tensor, alpha: float) -> Dict[str, float]:
    return {
        ans: patch_full_logprob_component(
            model, tokenizer, device, prompt, ans, layer_idx, component, patch_pos, patch_vec, alpha
        )
        for ans in candidates
    }


def run_model(args):
    model, tokenizer, device = load_model_flash(args.model)
    try:
        info = get_model_info(model, args.model)
        layers = selected_layers(info.n_layers)
        probe_layers = [layers[-2], layers[-1]] if len(layers) >= 2 else [layers[-1]]
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
            bp = base_pos.get("prompt_last")
            rp = repair_pos.get("prompt_last")
            if bp is None or rp is None:
                continue
            base_out = collect_component_outputs(model, tokenizer, device, case["base_prompt"], probe_layers, components)
            repair_out = collect_component_outputs(model, tokenizer, device, case["repair_prompt"], probe_layers, components)
            for comp in components:
                for li in probe_layers:
                    if li not in base_out.get(comp, {}) or li not in repair_out.get(comp, {}):
                        continue
                    if bp >= base_out[comp][li].shape[1] or rp >= repair_out[comp][li].shape[1]:
                        continue
                    delta = repair_out[comp][li][0, rp, :] - base_out[comp][li][0, bp, :]
                    rnd = random_same_norm(delta, seed=si * 1231 + li * 17 + len(comp))
                    for mode, vec in [("repair_delta", delta), ("random_same_norm", rnd)]:
                        patched = winner_stats(
                            patched_score_map(
                                model, tokenizer, device, case["base_prompt"], values,
                                li, comp, bp, vec, args.alpha
                            ),
                            correct,
                        )
                        old_wrong = base["top_wrong"]
                        rows.append({
                            "sample_idx": si,
                            "case": {k: case[k] for k in ["tt_idx", "object", "relation", "category", "correct"]},
                            "target_case": target_case,
                            "component": comp,
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
            "phase": 589,
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
        key = f"{r['component']}|L{r['layer']}|{r['mode']}"
        item = by_key.setdefault(key, {
            "component": r["component"],
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
    )[:12]
    for item in best:
        log(
            f"  {item['component']} L{item['layer']} {item['mode']}: "
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
        args.max_samples = 4
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
    out_path = out_dir / f"phase589_{args.model}_component_value_path_attribution{suffix}.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    log(f"Wrote {out_path}")
    log(f"Total time: {result['total_time_min']} min")

    if args.hard_exit_after_model:
        os._exit(0)


if __name__ == "__main__":
    main()
