#!/usr/bin/env python3
"""
Phase 586: Distributed Relation-Filter Path Patch
分布式关系过滤路径修补

Phase 585 found:
  - polarity-format gate can be repaired by answer/prompt-last hidden delta.
  - value relation-filter gate cannot be repaired by the same local delta.

Phase 586 tests whether value repair lives at relation/category/rule positions.
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
from typing import Dict, List, Optional, Tuple

import torch

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from model_utils import get_layers, get_model_info, release_model  # noqa: E402
from phase584_gate_repair import (  # noqa: E402
    CANDIDATE_CATEGORIES,
    CANDIDATE_OBJECTS,
    CANDIDATE_RELATIONS,
    CANDIDATE_VALUES,
    build_cat_rel_truth_tables,
    build_gold_cat_prompt,
    build_oc_truth_tables,
    compute_full_string_logprob_batch,
    load_model_flash,
)

OUT_ROOT = Path("results/glm5_phase586_distributed_value_path_patch")


def log(msg: str = "") -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def selected_layers(n_layers: int) -> List[int]:
    raw = [n_layers // 4, n_layers // 2, (3 * n_layers) // 4, n_layers - 2]
    return sorted(set(max(0, min(n_layers - 1, x)) for x in raw))


def score_prompt(model, tokenizer, device, prompt: str, candidates: List[str], correct: str) -> Dict:
    scores = compute_full_string_logprob_batch(model, tokenizer, device, prompt, candidates)
    pred = max(scores, key=lambda k: scores[k][0])
    sorted_scores = sorted((v[0], k) for k, v in scores.items())
    margin = sorted_scores[-1][0] - sorted_scores[-2][0] if len(sorted_scores) >= 2 else 0.0
    return {
        "pred": pred,
        "correct": pred == correct,
        "margin": margin,
        "correct_logprob": scores[correct][0],
        "scores": {k: v[0] for k, v in scores.items()},
    }


def token_pos_after_substring(tokenizer, prompt: str, needle: str, occurrence: str = "last") -> Optional[int]:
    if occurrence == "first":
        idx = prompt.find(needle)
    else:
        idx = prompt.rfind(needle)
    if idx < 0:
        return None
    prefix = prompt[:idx + len(needle)]
    pos = len(tokenizer.encode(prefix, add_special_tokens=False)) - 1
    return max(0, pos)


def get_hidden(model, tokenizer, device, prompt: str, layer_indices: List[int]) -> Dict[int, torch.Tensor]:
    input_ids = torch.tensor([tokenizer.encode(prompt, add_special_tokens=False)], device=device)
    with torch.inference_mode():
        out = model(input_ids=input_ids, output_hidden_states=True, return_dict=True)
    result = {}
    for li in layer_indices:
        hs_idx = li + 1
        if hs_idx < len(out.hidden_states):
            result[li] = out.hidden_states[hs_idx][0].detach().float().cpu()
    return result


def patch_full_logprob(model, tokenizer, device, prompt: str, answer: str, layer_idx: int,
                       patch_pos: int, patch_vec: torch.Tensor, mode: str, alpha: float) -> Tuple[float, list]:
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    answer_ids = tokenizer.encode(" " + answer, add_special_tokens=False)
    if not answer_ids:
        answer_ids = tokenizer.encode(answer, add_special_tokens=False)
    if not answer_ids:
        return -100.0, []
    all_ids = prompt_ids + answer_ids
    if patch_pos < 0 or patch_pos >= len(prompt_ids):
        return -100.0, []

    layers = get_layers(model)
    target = layers[layer_idx]
    vec = patch_vec.to(device=device)

    def hook(_module, _inputs, output):
        if isinstance(output, tuple):
            h = output[0]
            h_new = h.clone()
            if mode == "replace":
                h_new[0, patch_pos, :] = vec.to(dtype=h_new.dtype)
            else:
                h_new[0, patch_pos, :] = h_new[0, patch_pos, :] + alpha * vec.to(dtype=h_new.dtype)
            return (h_new,) + output[1:]
        h_new = output.clone()
        if mode == "replace":
            h_new[0, patch_pos, :] = vec.to(dtype=h_new.dtype)
        else:
            h_new[0, patch_pos, :] = h_new[0, patch_pos, :] + alpha * vec.to(dtype=h_new.dtype)
        return h_new

    handle = target.register_forward_hook(hook)
    try:
        total = 0.0
        per_token = []
        with torch.inference_mode():
            full_input = torch.tensor([all_ids], device=device)
            out = model(input_ids=full_input, return_dict=True)
            logits = out.logits[0].float()
            start = len(prompt_ids) - 1
            for i, tid in enumerate(answer_ids):
                pos = start + i
                if pos >= logits.shape[0]:
                    break
                lp = torch.log_softmax(logits[pos], dim=-1)[tid]
                val = float(lp.cpu())
                total += val
                per_token.append(val)
        return total, per_token
    finally:
        handle.remove()


def patched_score(model, tokenizer, device, prompt: str, candidates: List[str], correct: str,
                  layer_idx: int, patch_pos: int, patch_vec: torch.Tensor,
                  mode: str, alpha: float) -> Dict:
    scores = {
        ans: patch_full_logprob(model, tokenizer, device, prompt, ans, layer_idx, patch_pos, patch_vec, mode, alpha)
        for ans in candidates
    }
    pred = max(scores, key=lambda k: scores[k][0])
    sorted_scores = sorted((v[0], k) for k, v in scores.items())
    margin = sorted_scores[-1][0] - sorted_scores[-2][0] if len(sorted_scores) >= 2 else 0.0
    return {
        "pred": pred,
        "correct": pred == correct,
        "margin": margin,
        "correct_logprob": scores[correct][0],
    }


def random_same_norm(delta: torch.Tensor, seed: int) -> torch.Tensor:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    rnd = torch.randn(delta.shape, generator=gen, dtype=torch.float32)
    return rnd / torch.linalg.vector_norm(rnd).clamp_min(1e-8) * torch.linalg.vector_norm(delta).clamp_min(1e-8)


def build_filter_prompt(oc_table, crv_table, obj: str, rel: str, seed: int, wrong: bool = False):
    rng = random.Random(seed)
    correct_cat = oc_table[obj]
    use_rel = rel
    if wrong:
        candidates = [r for r in CANDIDATE_RELATIONS[:2] if r != rel]
        if candidates:
            use_rel = candidates[0]
    correct_val = crv_table[(correct_cat, rel)]
    crv_rules = list(crv_table.items())
    rng.shuffle(crv_rules)
    relevant = [((cat, r), val) for (cat, r), val in crv_rules if r == use_rel]
    prompt = "Rules:\n" + "\n".join([f"{cat} {r} {val}." for (cat, r), val in relevant])
    prompt += f"\n\n{obj} belongs to {correct_cat}."
    prompt += f"\nQuestion: {correct_cat} {rel} ?\nAnswer:"
    return prompt, correct_cat, correct_val, use_rel


def build_cases(n_tables: int, max_samples: int):
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
                cat = oc_table[obj]
                correct_val = crv_table[(cat, rel)]
                base_prompt, _, _ = build_gold_cat_prompt(None, oc_table, crv_table, obj, rel, seed=tt_idx * 100)
                repair_prompt, _, _, repair_rel = build_filter_prompt(oc_table, crv_table, obj, rel, seed=tt_idx * 100)
                wrong_prompt, _, _, wrong_rel = build_filter_prompt(oc_table, crv_table, obj, rel, seed=tt_idx * 100, wrong=True)
                yield {
                    "tt_idx": tt_idx,
                    "object": obj,
                    "relation": rel,
                    "category": cat,
                    "correct": correct_val,
                    "candidates": values,
                    "base_prompt": base_prompt,
                    "repair_prompt": repair_prompt,
                    "wrong_prompt": wrong_prompt,
                    "repair_rel": repair_rel,
                    "wrong_rel": wrong_rel,
                }
                count += 1
                if count >= max_samples:
                    return


def case_positions(tokenizer, case: Dict, prompt: str, relation_for_rule: str) -> Dict[str, Optional[int]]:
    cat = case["category"]
    rel = case["relation"]
    correct = case["correct"]
    return {
        "prompt_last": len(tokenizer.encode(prompt, add_special_tokens=False)) - 1,
        "query_relation": token_pos_after_substring(tokenizer, prompt, rel, "last"),
        "query_category": token_pos_after_substring(tokenizer, prompt, cat, "last"),
        "rule_relation": token_pos_after_substring(tokenizer, prompt, relation_for_rule, "first"),
        "rule_value": token_pos_after_substring(tokenizer, prompt, correct, "first"),
    }


def run_model(args):
    model, tokenizer, device = load_model_flash(args.model)
    try:
        info = get_model_info(model, args.model)
        layers = selected_layers(info.n_layers)
        cases = list(build_cases(args.n_tables, args.max_samples))
        log(f"{args.model}: n_layers={info.n_layers}, layers={layers}, cases={len(cases)}")
        rows = []
        for si, case in enumerate(cases):
            base = score_prompt(model, tokenizer, device, case["base_prompt"], case["candidates"], case["correct"])
            repair = score_prompt(model, tokenizer, device, case["repair_prompt"], case["candidates"], case["correct"])
            wrong = score_prompt(model, tokenizer, device, case["wrong_prompt"], case["candidates"], case["correct"])
            base_h = get_hidden(model, tokenizer, device, case["base_prompt"], layers)
            repair_h = get_hidden(model, tokenizer, device, case["repair_prompt"], layers)
            wrong_h = get_hidden(model, tokenizer, device, case["wrong_prompt"], layers)
            base_pos = case_positions(tokenizer, case, case["base_prompt"], case["relation"])
            repair_pos = case_positions(tokenizer, case, case["repair_prompt"], case["repair_rel"])
            wrong_pos = case_positions(tokenizer, case, case["wrong_prompt"], case["wrong_rel"])

            for pos_name in ["prompt_last", "query_relation", "query_category", "rule_relation", "rule_value"]:
                bp = base_pos.get(pos_name)
                rp = repair_pos.get(pos_name)
                wp = wrong_pos.get(pos_name)
                if bp is None or rp is None or wp is None:
                    continue
                for li in layers:
                    if bp >= base_h[li].shape[0] or rp >= repair_h[li].shape[0] or wp >= wrong_h[li].shape[0]:
                        continue
                    delta = repair_h[li][rp] - base_h[li][bp]
                    wrong_delta = wrong_h[li][wp] - base_h[li][bp]
                    rnd = random_same_norm(delta, seed=si * 1009 + li * 17 + len(pos_name))
                    for mode, vec in [
                        ("add_repair", delta),
                        ("add_wrong_relation", wrong_delta),
                        ("add_random", rnd),
                        ("replace_repair", repair_h[li][rp]),
                    ]:
                        score = patched_score(
                            model, tokenizer, device, case["base_prompt"], case["candidates"], case["correct"],
                            li, bp, vec, "replace" if mode == "replace_repair" else "add", args.alpha)
                        rows.append({
                            "sample_idx": si,
                            "layer": li,
                            "position": pos_name,
                            "mode": mode,
                            "case": {k: case[k] for k in ["tt_idx", "object", "relation", "category", "correct"]},
                            "base_correct": base["correct"],
                            "repair_correct": repair["correct"],
                            "wrong_prompt_correct": wrong["correct"],
                            "base_pred": base["pred"],
                            "patch_pred": score["pred"],
                            "patch_correct": score["correct"],
                            "patch_correct_logprob": score["correct_logprob"],
                            "base_correct_logprob": base["correct_logprob"],
                            "target_case": (not base["correct"]) and repair["correct"],
                        })

        summary = summarize(rows)
        return {
            "phase": 586,
            "model": args.model,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "n_layers": info.n_layers,
            "probe_layers": layers,
            "n_tables": args.n_tables,
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
        key = f"{r['position']}|L{r['layer']}|{r['mode']}"
        item = by_key.setdefault(key, {
            "position": r["position"],
            "layer": r["layer"],
            "mode": r["mode"],
            "n": 0,
            "base_correct": 0,
            "repair_correct": 0,
            "patch_correct": 0,
            "target_n": 0,
            "target_patch_correct": 0,
            "mean_logprob_gain": 0.0,
        })
        item["n"] += 1
        item["base_correct"] += int(r["base_correct"])
        item["repair_correct"] += int(r["repair_correct"])
        item["patch_correct"] += int(r["patch_correct"])
        item["mean_logprob_gain"] += r["patch_correct_logprob"] - r["base_correct_logprob"]
        if r["target_case"]:
            item["target_n"] += 1
            item["target_patch_correct"] += int(r["patch_correct"])
    for item in by_key.values():
        n = max(1, item["n"])
        tn = max(1, item["target_n"])
        item["base_accuracy"] = item["base_correct"] / n
        item["repair_accuracy"] = item["repair_correct"] / n
        item["patch_accuracy"] = item["patch_correct"] / n
        item["target_patch_accuracy"] = item["target_patch_correct"] / tn
        item["mean_logprob_gain"] /= n

    best = sorted(
        by_key.values(),
        key=lambda x: (x["target_n"], x["target_patch_accuracy"], x["patch_accuracy"], x["mean_logprob_gain"]),
        reverse=True,
    )[:12]
    for item in best:
        log(
            f"  {item['position']} L{item['layer']} {item['mode']}: "
            f"patch={item['patch_accuracy']:.3f}, target={item['target_patch_correct']}/{item['target_n']}, "
            f"gain={item['mean_logprob_gain']:.3f}"
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
    out_path = out_dir / f"phase586_{args.model}_distributed_value_path_patch{suffix}.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    log(f"Wrote {out_path}")
    log(f"Total time: {result['total_time_min']} min")

    if args.hard_exit_after_model:
        os._exit(0)


if __name__ == "__main__":
    main()
