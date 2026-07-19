#!/usr/bin/env python3
"""
Phase 965: 从直接EOS注入到自然停止控制图谱
=============================================
Phase 964实现首次strict-clean(delayed2_b=20)。本阶段：
Task 1: 大规模delayed注入验证 (20-50 prompts)
Task 2: 动态delay (检测句号/边界后注入)
Task 3: 上界曲线 CleanRate(b,d)
Task 4: 差分法自然组件联合ablate (多head同时消融)
"""

from __future__ import annotations
import gc, json, sys, time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from transformers import LogitsProcessor

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))
from model_utils import load_model, get_layers, get_model_info, release_model, get_W_U
from phase951_protocol_atlas import ensure_dir
from phase964_forward_diff import (
    log, get_head_dims, make_head_hook, make_eos_inject_hook, EN_PROMPTS_50,
    MAX_TOKENS, RESULT_DIR as RESULT_DIR964
)

PHASE = 965
RESULT_DIR = Path("results/phase965_stop_control")

# Per-model optimal injection params (from Phase 964)
MODEL_CONFIG = {
    "qwen3":      {"b": 30, "d": 2, "n_val": 50, "n_grid": 10, "n_dyn": 10},
    "glm4":       {"b": 20, "d": 2, "n_val": 15, "n_grid": 3,  "n_dyn": 5},
    "deepseek7b": {"b": 20, "d": 2, "n_val": 20, "n_grid": 5,  "n_dyn": 10},
}

EXPECTED = {
    "The capital of France is": "Paris", "The largest planet is": "Jupiter",
    "Water boils at": "100", "The speed of light is": "299",
    "The sun is a": "star", "Dogs are": "animal", "The sky is": "blue",
    "Grass is": "green", "Fire needs": "oxygen", "Ice is": "frozen",
    "The Earth is": "round", "A triangle has": "three",
    "Shakespeare was": "English", "Tokyo is the capital of": "Japan",
    "The Pacific Ocean is": "large", "Gold is a": "metal",
    "Plants need": "water", "Humans breathe": "oxygen",
    "The moon is": "natural", "Birds can": "fly",
    "The largest country is": "Russia", "A square has": "four",
    "Mathematics is": "study", "Iron is a": "metal",
    "Trees produce": "oxygen", "Stars are": "sun",
}


class DynamicEOSProcessor(LogitsProcessor):
    """Inject EOS bias after boundary token detected (dynamic delay)."""
    def __init__(self, eos_id, bias, min_delay=2, boundary_ids=None):
        self.eos_id = eos_id
        self.bias = bias
        self.min_delay = min_delay
        self.boundary_ids = set(boundary_ids or [])
        self.step = 0
        self.inject = False

    def __call__(self, input_ids, scores):
        if self.step >= self.min_delay:
            if len(input_ids[0]) > 0 and input_ids[0, -1].item() in self.boundary_ids:
                self.inject = True
            if self.inject:
                scores[:, self.eos_id] += self.bias
        self.step += 1
        return scores


def evaluate_clean_v2(prompt, generated, has_eos, n_tokens):
    """Extended strict-clean: strip special tokens before ASCII check."""
    expected = EXPECTED.get(prompt, "")
    # Strip common special tokens
    for special in ["<|endoftext|>", "<|im_end|>", "<｜end▁of▁sentence｜>",
                    "</think>", "<think>", "<|end|>", "\\boxed{}"]:
        generated_clean = generated.replace(special, "")
    # Check ASCII on cleaned content
    is_ascii = all(ord(c) < 256 for c in generated)
    is_ascii_clean = all(ord(c) < 256 for c in generated_clean) if 'generated_clean' in dir() else is_ascii
    is_short = 0 < n_tokens < 15
    has_expected = (expected == "") or (expected.lower() in generated.lower())
    strict_clean = has_eos and is_short and has_expected and is_ascii
    strict_clean_v2 = has_eos and is_short and has_expected and is_ascii_clean
    return {
        "strict_clean": strict_clean,
        "strict_clean_v2": strict_clean_v2,
        "has_eos": has_eos, "is_short": is_short,
        "has_expected": has_expected, "is_ascii": is_ascii, "is_ascii_clean": is_ascii_clean,
    }


def get_boundary_ids(tokenizer):
    """Get token IDs for boundary tokens (period, newline, comma)."""
    ids = set()
    for tok_str in [".", "\n", ",", "。", "，"]:
        toks = tokenizer.encode(tok_str, add_special_tokens=False)
        if toks:
            ids.add(toks[0])
    return ids


def run_model(model_name):
    log(f"\n{'='*60}")
    log(f"Phase 965: {model_name}")
    log(f"{'='*60}")

    model_dir = RESULT_DIR / model_name
    ensure_dir(model_dir)

    cfg = MODEL_CONFIG[model_name]
    model, tokenizer, device = load_model(model_name)
    info = get_model_info(model, model_name)
    n_heads, d_head = get_head_dims(model, info)
    layers = get_layers(model)
    n_layers = info.n_layers
    eos_id = tokenizer.eos_token_id
    boundary_ids = get_boundary_ids(tokenizer)
    log(f"  {info.model_class}, {info.n_layers}L  boundary_ids={boundary_ids}")

    results = {"model": model_name}
    t_start = time.time()

    # ============================================================
    # Task 1: Large-scale delayed injection validation
    # ============================================================
    log(f"  Task 1: Large-scale delayed injection (b={cfg['b']}, d={cfg['d']}, {cfg['n_val']} prompts)...")
    prompts = EN_PROMPTS_50[:cfg["n_val"]]
    all_r1 = []
    for pi, prompt in enumerate(prompts):
        # Normal
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        try:
            with torch.no_grad():
                oid = model.generate(input_ids, max_new_tokens=30, do_sample=False, pad_token_id=eos_id)
            gt = oid[0][input_ids.shape[1]:]
            gen_n = tokenizer.decode(gt, skip_special_tokens=False)
            he_n = gt[-1].item() == eos_id; ng_n = len(gt)
        except: gen_n = "ERROR"; he_n = False; ng_n = 0

        # Delayed injection
        h = model.lm_head.register_forward_hook(make_eos_inject_hook(eos_id, cfg["b"], cfg["d"]))
        try:
            with torch.no_grad():
                oid = model.generate(input_ids, max_new_tokens=30, do_sample=False, pad_token_id=eos_id)
            gt = oid[0][input_ids.shape[1]:]
            gen_i = tokenizer.decode(gt, skip_special_tokens=False)
            he_i = gt[-1].item() == eos_id; ng_i = len(gt)
        except: gen_i = "ERROR"; he_i = False; ng_i = 0
        h.remove()

        ce_n = evaluate_clean_v2(prompt, gen_n, he_n, ng_n)
        ce_i = evaluate_clean_v2(prompt, gen_i, he_i, ng_i)
        all_r1.append({
            "prompt": prompt, "normal_gen": gen_n[:100], "normal_eos": he_n, "normal_clean": ce_n["strict_clean"],
            "injected_gen": gen_i[:100], "injected_eos": he_i,
            "injected_clean": ce_i["strict_clean"], "injected_clean_v2": ce_i["strict_clean_v2"],
            "injected_tokens": ng_i,
        })
        if (pi + 1) % 10 == 0:
            log(f"    {pi+1}/{len(prompts)} prompts ({time.time()-t_start:.0f}s)")

    n = len(all_r1)
    normal_clean = sum(r["normal_clean"] for r in all_r1)
    injected_clean = sum(r["injected_clean"] for r in all_r1)
    injected_clean_v2 = sum(r["injected_clean_v2"] for r in all_r1)
    injected_eos = sum(r["injected_eos"] for r in all_r1)
    s1 = {
        "normal_clean_rate": normal_clean / n,
        "injected_clean_rate": injected_clean / n,
        "injected_clean_v2_rate": injected_clean_v2 / n,
        "injected_eos_rate": injected_eos / n,
        "n_prompts": n,
    }
    results["task1"] = {"task": "task1_large_scale", "config": cfg, "summary": s1, "raw_results": all_r1}
    (model_dir / "task1_large_scale.json").write_text(json.dumps(results["task1"], ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"    Normal clean: {normal_clean}/{n} = {s1['normal_clean_rate']:.2f}")
    log(f"    Injected clean: {injected_clean}/{n} = {s1['injected_clean_rate']:.2f}")
    log(f"    Injected clean_v2: {injected_clean_v2}/{n} = {s1['injected_clean_v2_rate']:.2f}")
    log(f"    Injected EOS: {injected_eos}/{n} = {s1['injected_eos_rate']:.2f}")
    # Print some samples
    for r in all_r1[:3]:
        log(f"    '{r['prompt'][:25]}': injected='{r['injected_gen'][:50]}' eos={r['injected_eos']} clean={r['injected_clean']}")
    log(f"    Task 1 done ({time.time()-t_start:.0f}s)")

    if torch.cuda.is_available(): torch.cuda.empty_cache()

    # ============================================================
    # Task 2: Dynamic delay (inject after boundary token)
    # ============================================================
    log(f"  Task 2: Dynamic delay (b={cfg['b']}, min_delay=2, boundary detection)...")
    prompts_dyn = EN_PROMPTS_50[:cfg["n_dyn"]]
    all_r2 = []
    for pi, prompt in enumerate(prompts_dyn):
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        processor = DynamicEOSProcessor(eos_id, cfg["b"], min_delay=2, boundary_ids=boundary_ids)
        try:
            with torch.no_grad():
                oid = model.generate(input_ids, max_new_tokens=30, do_sample=False,
                                     pad_token_id=eos_id, logits_processor=[processor])
            gt = oid[0][input_ids.shape[1]:]
            gen = tokenizer.decode(gt, skip_special_tokens=False)
            he = gt[-1].item() == eos_id; ng = len(gt)
        except Exception as e:
            gen = f"ERROR: {e}"; he = False; ng = 0
        ce = evaluate_clean_v2(prompt, gen, he, ng)
        all_r2.append({"prompt": prompt, "generated": gen[:100], "has_eos": he,
                       "n_tokens": ng, "strict_clean": ce["strict_clean"],
                       "strict_clean_v2": ce["strict_clean_v2"]})
        if (pi + 1) % 5 == 0:
            log(f"    {pi+1}/{len(prompts_dyn)} prompts")

    n2 = len(all_r2)
    dyn_clean = sum(r["strict_clean"] for r in all_r2)
    dyn_clean_v2 = sum(r["strict_clean_v2"] for r in all_r2)
    dyn_eos = sum(r["has_eos"] for r in all_r2)
    s2 = {"dynamic_clean_rate": dyn_clean / n2, "dynamic_clean_v2_rate": dyn_clean_v2 / n2,
          "dynamic_eos_rate": dyn_eos / n2, "n_prompts": n2}
    results["task2"] = {"task": "task2_dynamic_delay", "summary": s2, "raw_results": all_r2}
    (model_dir / "task2_dynamic_delay.json").write_text(json.dumps(results["task2"], ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"    Dynamic clean: {dyn_clean}/{n2} = {s2['dynamic_clean_rate']:.2f}")
    log(f"    Dynamic clean_v2: {dyn_clean_v2}/{n2} = {s2['dynamic_clean_v2_rate']:.2f}")
    log(f"    Dynamic EOS: {dyn_eos}/{n2} = {s2['dynamic_eos_rate']:.2f}")
    for r in all_r2[:3]:
        log(f"    '{r['prompt'][:25]}': gen='{r['generated'][:50]}' eos={r['has_eos']} clean={r['strict_clean']}")
    log(f"    Task 2 done ({time.time()-t_start:.0f}s)")

    if torch.cuda.is_available(): torch.cuda.empty_cache()

    # ============================================================
    # Task 3: Upper bound curve CleanRate(b, d)
    # ============================================================
    log(f"  Task 3: Upper bound curve CleanRate(b,d)...")
    if model_name == "qwen3":
        b_values = [20, 25, 30, 35, 40]
    elif model_name == "glm4":
        b_values = [10, 15, 20, 25, 30]
    else:
        b_values = [10, 15, 20, 25, 30]
    d_values = [2, 3]
    prompts_grid = EN_PROMPTS_50[:cfg["n_grid"]]

    grid_results = []
    for b in b_values:
        for d in d_values:
            clean_count = 0; eos_count = 0; total = 0
            for pi, prompt in enumerate(prompts_grid):
                input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
                h = model.lm_head.register_forward_hook(make_eos_inject_hook(eos_id, b, d))
                try:
                    with torch.no_grad():
                        oid = model.generate(input_ids, max_new_tokens=30, do_sample=False, pad_token_id=eos_id)
                    gt = oid[0][input_ids.shape[1]:]
                    gen = tokenizer.decode(gt, skip_special_tokens=False)
                    he = gt[-1].item() == eos_id; ng = len(gt)
                except: gen = "ERROR"; he = False; ng = 0
                h.remove()
                ce = evaluate_clean_v2(prompt, gen, he, ng)
                clean_count += int(ce["strict_clean"])
                eos_count += int(he)
                total += 1
            grid_results.append({
                "b": b, "d": d,
                "clean_rate": clean_count / total,
                "eos_rate": eos_count / total,
                "n": total,
            })
            log(f"    b={b:2d} d={d}: clean={clean_count}/{total}={clean_count/total:.2f}  eos={eos_count}/{total}={eos_count/total:.2f}")

    results["task3"] = {"task": "task3_upper_bound", "grid": grid_results}
    (model_dir / "task3_upper_bound.json").write_text(json.dumps(results["task3"], ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"    Task 3 done ({time.time()-t_start:.0f}s)")

    elapsed = time.time() - t_start
    results["elapsed_seconds"] = elapsed
    log(f"\n  Total: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    release_model(model)

    save_path = RESULT_DIR / f"{model_name}_result.json"
    save_path.write_text(json.dumps(results, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    log(f"  {model_name} complete. Saved: {save_path}")
    return results


def main():
    ensure_dir(RESULT_DIR)
    log(f"Phase {PHASE} started")
    log(f"Tasks: 1=large_scale, 2=dynamic_delay, 3=upper_bound_curve")

    model_name = sys.argv[1] if len(sys.argv) > 1 else None
    if model_name:
        run_model(model_name)
    else:
        for m in ["qwen3", "glm4", "deepseek7b"]:
            try: run_model(m)
            except Exception as e:
                log(f"  {m} FAILED: {e}"); import traceback; traceback.print_exc()
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()
    log(f"\nPhase {PHASE} complete!")


if __name__ == "__main__":
    main()
