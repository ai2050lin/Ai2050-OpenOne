#!/usr/bin/env python3
"""
Phase 585: State Gate Localization and Hidden Causal Repair
状态门定位与隐藏因果修复

Phase 584 showed prompt-level repair:
  - relation-filter repairs value retrieval.
  - rule+format repairs polarity readout.

Phase 585 asks whether the repair state leaves a hidden-state delta that can
causally repair the base prompt at selected layers.
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
from typing import Dict, Iterable, List, Tuple

import torch

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from model_utils import get_layers, get_model_info, release_model  # noqa: E402
from phase584_gate_repair import (  # noqa: E402
    ANSWER_FORMATS,
    CANDIDATE_CATEGORIES,
    CANDIDATE_OBJECTS,
    CANDIDATE_RELATIONS,
    CANDIDATE_VALUES,
    CATEGORY_NEGATIVES,
    build_cat_rel_truth_tables,
    build_gold_cat_prompt,
    build_oc_truth_tables,
    compute_full_string_logprob_batch,
    load_model_flash,
)

OUT_ROOT = Path("results/glm5_phase585_state_gate_hidden_patch")


def log(msg: str = "") -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def candidate_prediction(scores: Dict[str, Tuple[float, list]]) -> Tuple[str, float]:
    pred = max(scores, key=lambda k: scores[k][0])
    sorted_scores = sorted((v[0], k) for k, v in scores.items())
    if len(sorted_scores) < 2:
        return pred, 0.0
    return pred, sorted_scores[-1][0] - sorted_scores[-2][0]


def score_prompt(model, tokenizer, device, prompt: str, candidates: List[str], correct: str) -> Dict:
    scores = compute_full_string_logprob_batch(model, tokenizer, device, prompt, candidates)
    pred, margin = candidate_prediction(scores)
    return {
        "pred": pred,
        "correct": pred == correct,
        "margin": margin,
        "correct_logprob": scores[correct][0],
        "scores": {k: v[0] for k, v in scores.items()},
    }


def get_hidden_vectors(model, tokenizer, device, prompt: str, layer_indices: List[int]) -> Dict[int, torch.Tensor]:
    input_ids = torch.tensor([tokenizer.encode(prompt, add_special_tokens=False)], device=device)
    with torch.inference_mode():
        out = model(input_ids=input_ids, output_hidden_states=True, return_dict=True)
    vectors = {}
    for li in layer_indices:
        hs_idx = li + 1
        if hs_idx < len(out.hidden_states):
            vectors[li] = out.hidden_states[hs_idx][0, -1, :].detach().float().cpu()
    return vectors


def patched_full_string_logprob(model, tokenizer, device, prompt: str, answer_str: str,
                                layer_idx: int, delta_vec: torch.Tensor, alpha: float) -> Tuple[float, list]:
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    answer_token_ids = tokenizer.encode(" " + answer_str, add_special_tokens=False)
    if not answer_token_ids:
        answer_token_ids = tokenizer.encode(answer_str, add_special_tokens=False)
    if not answer_token_ids:
        return -100.0, []
    all_token_ids = prompt_ids + answer_token_ids
    patch_pos = len(prompt_ids) - 1
    if patch_pos < 0:
        return -100.0, []

    layers = get_layers(model)
    target = layers[layer_idx]
    delta = delta_vec.to(device=device)

    def hook(_module, _inputs, output):
        if isinstance(output, tuple):
            h = output[0]
            h_new = h.clone()
            h_new[0, patch_pos, :] = h_new[0, patch_pos, :] + alpha * delta.to(dtype=h_new.dtype)
            return (h_new,) + output[1:]
        h_new = output.clone()
        h_new[0, patch_pos, :] = h_new[0, patch_pos, :] + alpha * delta.to(dtype=h_new.dtype)
        return h_new

    handle = target.register_forward_hook(hook)
    try:
        total_logprob = 0.0
        per_token_logprobs = []
        with torch.inference_mode():
            full_input = torch.tensor([all_token_ids], device=device)
            outputs = model(input_ids=full_input, return_dict=True)
            logits = outputs.logits[0].float()
            answer_start = len(prompt_ids) - 1
            for i, tid in enumerate(answer_token_ids):
                pos = answer_start + i
                if pos >= logits.shape[0]:
                    break
                log_probs = torch.log_softmax(logits[pos], dim=-1)
                lp = float(log_probs[tid].cpu())
                total_logprob += lp
                per_token_logprobs.append(lp)
        return total_logprob, per_token_logprobs
    finally:
        handle.remove()


def patched_scores(model, tokenizer, device, prompt: str, candidates: List[str],
                   correct: str, layer_idx: int, delta_vec: torch.Tensor, alpha: float) -> Dict:
    scores = {
        ans: patched_full_string_logprob(
            model, tokenizer, device, prompt, ans, layer_idx, delta_vec, alpha)
        for ans in candidates
    }
    pred, margin = candidate_prediction(scores)
    return {
        "pred": pred,
        "correct": pred == correct,
        "margin": margin,
        "correct_logprob": scores[correct][0],
        "scores": {k: v[0] for k, v in scores.items()},
    }


def random_same_norm(delta: torch.Tensor, seed: int) -> torch.Tensor:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    rnd = torch.randn(delta.shape, generator=gen, dtype=torch.float32)
    rnd_norm = torch.linalg.vector_norm(rnd).clamp_min(1e-8)
    delta_norm = torch.linalg.vector_norm(delta).clamp_min(1e-8)
    return rnd / rnd_norm * delta_norm


def selected_layers(n_layers: int) -> List[int]:
    raw = [n_layers // 4, n_layers // 2, (3 * n_layers) // 4, n_layers - 2]
    return sorted(set(max(0, min(n_layers - 1, x)) for x in raw))


def build_relation_filter_prompt(oc_table, crv_table, obj: str, rel: str, seed: int) -> Tuple[str, str, str]:
    rng = random.Random(seed)
    correct_cat = oc_table[obj]
    correct_val = crv_table[(correct_cat, rel)]
    crv_rules = list(crv_table.items())
    rng.shuffle(crv_rules)
    relevant = [((cat, r), val) for (cat, r), val in crv_rules if r == rel]
    prompt = "Rules:\n" + "\n".join([f"{cat} {r} {val}." for (cat, r), val in relevant])
    prompt += f"\n\n{obj} belongs to {correct_cat}."
    prompt += f"\nQuestion: {correct_cat} {rel} ?\nAnswer:"
    return prompt, correct_cat, correct_val


def iter_value_cases(n_tables: int, max_samples: int):
    objects = CANDIDATE_OBJECTS[:8]
    categories = CANDIDATE_CATEGORIES[:4]
    relations = CANDIDATE_RELATIONS[:2]
    values = CANDIDATE_VALUES[:4]
    oc_tables = build_oc_truth_tables(objects, categories, n_tables)
    count = 0
    for tt_idx, oc_table in enumerate(oc_tables):
        crv_table = build_cat_rel_truth_tables(categories, relations, values, seed=tt_idx * 200)
        for obj in objects:
            for rel in relations:
                if obj not in oc_table:
                    continue
                correct_cat = oc_table[obj]
                if (correct_cat, rel) not in crv_table:
                    continue
                gold_prompt, _, correct_val = build_gold_cat_prompt(
                    None, oc_table, crv_table, obj, rel, seed=tt_idx * 100)
                repair_prompt, _, _ = build_relation_filter_prompt(
                    oc_table, crv_table, obj, rel, seed=tt_idx * 100)
                yield {
                    "tt_idx": tt_idx,
                    "object": obj,
                    "relation": rel,
                    "correct_cat": correct_cat,
                    "correct": correct_val,
                    "candidates": values,
                    "base_prompt": gold_prompt,
                    "repair_prompt": repair_prompt,
                }
                count += 1
                if count >= max_samples:
                    return


def best_polarity_format(model_name: str) -> str:
    return {"qwen3": "english", "glm4": "double", "deepseek7b": "english"}.get(model_name, "english")


def iter_polarity_cases(model_name: str, max_samples: int):
    fmt = best_polarity_format(model_name)
    yes_token, no_token = ANSWER_FORMATS[fmt]
    count = 0
    for cat, negatives in CATEGORY_NEGATIVES.items():
        for obj in negatives:
            alt_cat = "动物" if cat != "动物" else "水果"
            base_prompt = f"Question: {obj}是不是{cat}？\nAnswer:"
            repair_prompt = f"Rules:\n{obj} 属于 {alt_cat}, 不属于 {cat}.\n\nQuestion: {obj}是不是{cat}？\nAnswer:"
            yield {
                "object": obj,
                "category": cat,
                "format": fmt,
                "correct": no_token,
                "candidates": [yes_token, no_token],
                "base_prompt": base_prompt,
                "repair_prompt": repair_prompt,
            }
            count += 1
            if count >= max_samples:
                return


def run_gate_patch(model, tokenizer, device, gate: str, cases: Iterable[Dict],
                   layers: List[int], alpha: float) -> Dict:
    log(f"--- {gate} hidden patch ---")
    rows = []
    for sample_idx, case in enumerate(cases):
        candidates = case["candidates"]
        correct = case["correct"]
        base = score_prompt(model, tokenizer, device, case["base_prompt"], candidates, correct)
        repair = score_prompt(model, tokenizer, device, case["repair_prompt"], candidates, correct)
        base_h = get_hidden_vectors(model, tokenizer, device, case["base_prompt"], layers)
        repair_h = get_hidden_vectors(model, tokenizer, device, case["repair_prompt"], layers)

        for li in layers:
            delta = repair_h[li] - base_h[li]
            patched = patched_scores(
                model, tokenizer, device, case["base_prompt"], candidates, correct, li, delta, alpha)
            rnd = random_same_norm(delta, seed=sample_idx * 1009 + li)
            random_patched = patched_scores(
                model, tokenizer, device, case["base_prompt"], candidates, correct, li, rnd, alpha)
            rows.append({
                "sample_idx": sample_idx,
                "gate": gate,
                "layer": li,
                "alpha": alpha,
                "case": {k: v for k, v in case.items() if not k.endswith("_prompt") and k != "candidates"},
                "base_pred": base["pred"],
                "base_correct": base["correct"],
                "base_margin": base["margin"],
                "base_correct_logprob": base["correct_logprob"],
                "repair_pred": repair["pred"],
                "repair_correct": repair["correct"],
                "repair_margin": repair["margin"],
                "repair_correct_logprob": repair["correct_logprob"],
                "patch_pred": patched["pred"],
                "patch_correct": patched["correct"],
                "patch_margin": patched["margin"],
                "patch_correct_logprob": patched["correct_logprob"],
                "random_pred": random_patched["pred"],
                "random_correct": random_patched["correct"],
                "random_margin": random_patched["margin"],
                "random_correct_logprob": random_patched["correct_logprob"],
                "delta_norm": float(torch.linalg.vector_norm(delta)),
            })
    return summarize_rows(rows)


def summarize_rows(rows: List[Dict]) -> Dict:
    by_layer = {}
    for r in rows:
        key = f"L{r['layer']}"
        item = by_layer.setdefault(key, {
            "n": 0,
            "base_correct": 0,
            "repair_correct": 0,
            "patch_correct": 0,
            "random_correct": 0,
            "target_n": 0,
            "target_patch_correct": 0,
            "target_random_correct": 0,
            "mean_patch_logprob_gain": 0.0,
            "mean_random_logprob_gain": 0.0,
            "mean_delta_norm": 0.0,
        })
        item["n"] += 1
        item["base_correct"] += int(r["base_correct"])
        item["repair_correct"] += int(r["repair_correct"])
        item["patch_correct"] += int(r["patch_correct"])
        item["random_correct"] += int(r["random_correct"])
        item["mean_patch_logprob_gain"] += r["patch_correct_logprob"] - r["base_correct_logprob"]
        item["mean_random_logprob_gain"] += r["random_correct_logprob"] - r["base_correct_logprob"]
        item["mean_delta_norm"] += r["delta_norm"]
        if (not r["base_correct"]) and r["repair_correct"]:
            item["target_n"] += 1
            item["target_patch_correct"] += int(r["patch_correct"])
            item["target_random_correct"] += int(r["random_correct"])

    for item in by_layer.values():
        n = max(1, item["n"])
        item["base_accuracy"] = item["base_correct"] / n
        item["repair_accuracy"] = item["repair_correct"] / n
        item["patch_accuracy"] = item["patch_correct"] / n
        item["random_accuracy"] = item["random_correct"] / n
        item["mean_patch_logprob_gain"] /= n
        item["mean_random_logprob_gain"] /= n
        item["mean_delta_norm"] /= n
        tn = max(1, item["target_n"])
        item["target_patch_accuracy"] = item["target_patch_correct"] / tn
        item["target_random_accuracy"] = item["target_random_correct"] / tn

    for key, item in sorted(by_layer.items()):
        log(
            f"  {key}: base={item['base_accuracy']:.3f}, repair={item['repair_accuracy']:.3f}, "
            f"patch={item['patch_accuracy']:.3f}, random={item['random_accuracy']:.3f}, "
            f"target={item['target_patch_correct']}/{item['target_n']} "
            f"rnd={item['target_random_correct']}/{item['target_n']}"
        )
    return {"n_rows": len(rows), "by_layer": by_layer, "rows": rows}


def run_model(args):
    model, tokenizer, device = load_model_flash(args.model)
    try:
        info = get_model_info(model, args.model)
        layers = selected_layers(info.n_layers)
        log(f"{args.model}: n_layers={info.n_layers}, probe_layers={layers}")
        value_cases = list(iter_value_cases(args.n_tables, args.max_value_samples))
        polarity_cases = list(iter_polarity_cases(args.model, args.max_polarity_samples))
        result = {
            "phase": 585,
            "model": args.model,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "n_layers": info.n_layers,
            "probe_layers": layers,
            "n_tables": args.n_tables,
            "alpha": args.alpha,
            "n_value_cases": len(value_cases),
            "n_polarity_cases": len(polarity_cases),
        }
        result["value_relation_filter_gate"] = run_gate_patch(
            model, tokenizer, device, "value_relation_filter_gate", value_cases, layers, args.alpha)
        result["polarity_format_gate"] = run_gate_patch(
            model, tokenizer, device, "polarity_format_gate", polarity_cases, layers, args.alpha)
        return result
    finally:
        release_model(model)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--n-tables", type=int, default=6)
    parser.add_argument("--max-value-samples", type=int, default=24)
    parser.add_argument("--max-polarity-samples", type=int, default=24)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--output-dir", default=str(OUT_ROOT))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--confirm", action="store_true")
    parser.add_argument("--hard-exit-after-model", action="store_true")
    args = parser.parse_args()

    if args.smoke:
        args.n_tables = 2
        args.max_value_samples = 6
        args.max_polarity_samples = 6
        log("SMOKE TEST MODE")
    elif args.confirm:
        args.n_tables = max(args.n_tables, 8)
        args.max_value_samples = max(args.max_value_samples, 32)
        args.max_polarity_samples = max(args.max_polarity_samples, 30)
        log("CONFIRMATION TEST MODE")

    t0 = time.time()
    result = run_model(args)
    result["total_time_min"] = round((time.time() - t0) / 60.0, 2)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_smoke" if args.smoke else ("_confirm" if args.confirm else "")
    out_path = out_dir / f"phase585_{args.model}_state_gate_hidden_patch{suffix}.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    log(f"Wrote {out_path}")
    log(f"Total time: {result['total_time_min']} min")

    if args.hard_exit_after_model:
        os._exit(0)


if __name__ == "__main__":
    main()
