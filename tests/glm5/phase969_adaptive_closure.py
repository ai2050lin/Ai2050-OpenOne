#!/usr/bin/env python3
"""
Phase 969: 大规模自适应闭合与自然组件替代审计
==============================================
Phase 968: GLM4 adaptive b=100%(5p). Phase 969: 大规模验证+dynamic delay组合。

Task 1: GLM4 adaptive b大规模验证(20 prompts) — 确认>80%
Task 2: GLM4 adaptive b + dynamic delay(10 prompts) — 完成检测+自适应b
Task 3: DS7B adaptive b(20 prompts) — 跨模型验证
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
from model_utils import load_model, get_layers, get_model_info, release_model
from phase951_protocol_atlas import ensure_dir
from phase966_natural_stop import log, register_multi_head_ablation, get_boundary_ids
from phase964_forward_diff import make_head_hook, make_eos_inject_hook, get_head_dims, EN_PROMPTS_50

PHASE = 969
RESULT_DIR = Path("results/phase969_adaptive_closure")

SAFE_HEADS = {
    "glm4": [(38, 5), (39, 23), (38, 0), (35, 1)],
    "deepseek7b": [(27, 12), (25, 20), (26, 0), (26, 9), (26, 17), (26, 20), (25, 16), (26, 7), (26, 12), (25, 21)],
}

# Extended semantic equivalence
SEMANTIC_EQUIV = {
    "The capital of France is": ["paris"], "The largest planet is": ["jupiter"],
    "Water boils at": ["100", "212", "celsius", "fahrenheit", "boiling"],
    "The speed of light is": ["299", "300", "186", "light", "c"],
    "The sun is a": ["star", "sun"], "Dogs are": ["animal", "mammal", "loyal", "companion", "pet", "dog"],
    "The sky is": ["blue", "clear", "up", "atmosphere"],
    "Grass is": ["green", "plant"], "Fire needs": ["oxygen", "air", "fuel", "heat"],
    "Ice is": ["frozen", "solid", "water", "cold"],
    "The Earth is": ["round", "sphere", "planet", "earth"],
    "A triangle has": ["three", "3", "side", "angle"],
    "Shakespeare was": ["english", "playwright", "poet", "writer", "bard"],
    "Tokyo is the capital of": ["japan"], "The Pacific Ocean is": ["large", "biggest", "largest", "deep"],
    "Gold is a": ["metal", "element", "precious"], "Plants need": ["water", "sun", "light"],
    "Humans breathe": ["oxygen", "air"], "The moon is": ["natural", "satellite", "moon"],
    "Birds can": ["fly"], "The largest country is": ["russia"],
    "A square has": ["four", "4", "side"], "Mathematics is": ["study", "science", "abstract"],
    "Iron is a": ["metal", "element"], "Trees produce": ["oxygen", "fruit", "wood"],
    "Stars are": ["sun", "hot", "bright", "gas", "star"],
    "The brain is": ["organ", "mind", "think"], "Rivers flow": ["water", "down", "sea"],
    "Volcanoes erupt": ["lava", "magma", "fire"], "The heart pumps": ["blood"],
    "DNA contains": ["gene", "code", "information"], "Gravity pulls": ["down", "attract"],
    "Light travels": ["fast", "speed", "wave"], "Sound is": ["wave", "vibration", "noise"],
    "Heat is": ["energy", "thermal"], "A compass points": ["north", "direction"],
    "The equator is": ["line", "middle", "earth"], "Antarctica is": ["cold", "ice", "continent"],
    "Diamonds are": ["hard", "carbon", "gem"], "Oxygen is": ["gas", "element", "breath"],
    "The kidney filters": ["blood", "waste"], "Whales are": ["mammal", "sea", "large"],
    "The alphabet has": ["letter", "26"], "A century is": ["100", "year"],
    "The constitution is": ["law", "document", "rule"], "Bridges connect": ["place", "side"],
    "Computers process": ["data", "information"], "Languages evolve": ["change", "time"],
    "The internet is": ["network", "web", "online"],
}

SPECIAL_TOKENS = ["<|endoftext|>", "<|im_end|>", "<｜end▁of▁sentence｜>",
                  "</s>", "<|end|>", "</think>", "<think>", "\\boxed{}"]


def evaluate_clean_v4(prompt, generated, has_eos, n_tokens):
    """v4: extended semantic equivalence + special token stripping."""
    content = generated
    for st in SPECIAL_TOKENS:
        content = content.replace(st, "")
    content = content.strip()
    is_ascii = all(ord(c) < 256 for c in content)
    is_short = 0 < n_tokens < 15
    equiv = SEMANTIC_EQUIV.get(prompt, [])
    has_expected = len(equiv) == 0 or any(e.lower() in content.lower() for e in equiv)
    return {"strict_clean": has_eos and is_short and has_expected and is_ascii,
            "has_eos": has_eos, "is_short": is_short, "has_expected": has_expected,
            "is_ascii": is_ascii, "content": content[:80]}


class AdaptiveDynamicEOSProcessor(LogitsProcessor):
    """Dynamic delay + adaptive b: inject after boundary detected."""
    def __init__(self, eos_id, bias, min_delay=2, boundary_ids=None):
        self.eos_id = eos_id; self.bias = bias
        self.min_delay = min_delay
        self.boundary_ids = set(boundary_ids or [])
        self.step = 0; self.inject = False
    def __call__(self, input_ids, scores):
        if self.step >= self.min_delay:
            if len(input_ids[0]) > 0 and input_ids[0, -1].item() in self.boundary_ids:
                self.inject = True
            if self.inject:
                scores[:, self.eos_id] += self.bias
        self.step += 1
        return scores


def measure_gap(model, input_ids, eos_id, layers, safe_heads, d_head):
    """Measure base gap and ablated gap for a prompt."""
    with torch.no_grad():
        bl = model(input_ids, use_cache=False).logits[0, -1].detach().float().cpu().numpy()
    base_gap = float(np.sort(bl)[-1]) - (float(bl[eos_id]) if eos_id else 0)
    handles = register_multi_head_ablation(layers, safe_heads, d_head)
    with torch.no_grad():
        al = model(input_ids, use_cache=False).logits[0, -1].detach().float().cpu().numpy()
    for h in handles: h.remove()
    ablate_gap = float(np.sort(al)[-1]) - (float(al[eos_id]) if eos_id else 0)
    return base_gap, ablate_gap


def run_model(model_name):
    log(f"\n{'='*60}\nPhase 969: {model_name}\n{'='*60}")
    model_dir = RESULT_DIR / model_name
    ensure_dir(model_dir)
    model, tokenizer, device = load_model(model_name)
    info = get_model_info(model, model_name)
    n_heads, d_head = get_head_dims(model, info)
    layers = get_layers(model)
    eos_id = tokenizer.eos_token_id
    safe_heads = SAFE_HEADS.get(model_name, [])
    boundary_ids = get_boundary_ids(tokenizer)
    log(f"  {info.model_class}, {info.n_layers}L  safe_heads={len(safe_heads)}")

    results = {"model": model_name}
    t_start = time.time()

    if model_name == "glm4":
        # ============================================================
        # Task 1: GLM4 adaptive b large-scale (20 prompts)
        # ============================================================
        log(f"  Task 1: GLM4 adaptive b large-scale (20 prompts)...")
        prompts = EN_PROMPTS_50[:20]
        all_r1 = []
        for pi, prompt in enumerate(prompts):
            input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
            base_gap, ablate_gap = measure_gap(model, input_ids, eos_id, layers, safe_heads, d_head)
            b_adaptive = int(ablate_gap + 2)
            handles = register_multi_head_ablation(layers, safe_heads, d_head)
            h_inj = model.lm_head.register_forward_hook(make_eos_inject_hook(eos_id, b_adaptive, 2))
            try:
                with torch.no_grad():
                    oid = model.generate(input_ids, max_new_tokens=30, do_sample=False, pad_token_id=eos_id)
                gt = oid[0][input_ids.shape[1]:]
                gen = tokenizer.decode(gt, skip_special_tokens=False)
                he = gt[-1].item()==eos_id; ng = len(gt)
            except: gen="ERROR"; he=False; ng=0
            for h in handles: h.remove()
            h_inj.remove()
            ce = evaluate_clean_v4(prompt, gen, he, ng)
            all_r1.append({"prompt": prompt, "base_gap": base_gap, "ablate_gap": ablate_gap,
                           "b": b_adaptive, "generated": gen[:80], "has_eos": he,
                           "n_tokens": ng, "strict_clean": ce["strict_clean"],
                           "has_expected": ce["has_expected"]})
            if (pi+1) % 5 == 0: log(f"    {pi+1}/20 ({time.time()-t_start:.0f}s)")

        n = len(all_r1)
        clean = sum(r["strict_clean"] for r in all_r1)
        eos = sum(r["has_eos"] for r in all_r1)
        expected = sum(r["has_expected"] for r in all_r1)
        s1 = {"clean_rate": clean/n, "eos_rate": eos/n, "expected_rate": expected/n,
              "mean_b": float(np.mean([r["b"] for r in all_r1])), "n": n}
        results["task1"] = {"summary": s1, "raw_results": all_r1}
        (model_dir / "task1_large_scale.json").write_text(json.dumps(results["task1"], ensure_ascii=False, indent=2), encoding="utf-8")
        log(f"    Clean: {clean}/{n}={s1['clean_rate']:.2f}  EOS: {s1['eos_rate']:.2f}  Expected: {s1['expected_rate']:.2f}  mean_b={s1['mean_b']:.1f}")
        log(f"    Successes:")
        for r in all_r1:
            if r["strict_clean"]: log(f"      '{r['prompt'][:25]}': b={r['b']} '{r['generated'][:40]}'")
        log(f"    Failures (EOS but not clean):")
        for r in all_r1:
            if r["has_eos"] and not r["strict_clean"]: log(f"      '{r['prompt'][:25]}': b={r['b']} '{r['generated'][:40]}' exp={r['has_expected']}")
        log(f"    Task 1 done ({time.time()-t_start:.0f}s)")

        if torch.cuda.is_available(): torch.cuda.empty_cache()

        # ============================================================
        # Task 2: adaptive b + dynamic delay (10 prompts)
        # ============================================================
        log(f"  Task 2: adaptive b + dynamic delay (10 prompts)...")
        prompts2 = EN_PROMPTS_50[:10]
        all_r2 = []
        for pi, prompt in enumerate(prompts2):
            input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
            base_gap, ablate_gap = measure_gap(model, input_ids, eos_id, layers, safe_heads, d_head)
            b_adaptive = int(ablate_gap + 2)
            handles = register_multi_head_ablation(layers, safe_heads, d_head)
            proc = AdaptiveDynamicEOSProcessor(eos_id, b_adaptive, min_delay=2, boundary_ids=boundary_ids)
            try:
                with torch.no_grad():
                    oid = model.generate(input_ids, max_new_tokens=30, do_sample=False,
                                         pad_token_id=eos_id, logits_processor=[proc])
                gt = oid[0][input_ids.shape[1]:]
                gen = tokenizer.decode(gt, skip_special_tokens=False)
                he = gt[-1].item()==eos_id; ng = len(gt)
            except: gen="ERROR"; he=False; ng=0
            for h in handles: h.remove()
            ce = evaluate_clean_v4(prompt, gen, he, ng)
            all_r2.append({"prompt": prompt, "b": b_adaptive, "generated": gen[:80],
                           "has_eos": he, "n_tokens": ng, "strict_clean": ce["strict_clean"]})
            log(f"    p{pi}: b={b_adaptive} eos={he} clean={ce['strict_clean']} text={gen[:40]}")

        n2 = len(all_r2)
        clean2 = sum(r["strict_clean"] for r in all_r2)
        s2 = {"clean_rate": clean2/n2, "eos_rate": sum(r["has_eos"] for r in all_r2)/n2, "n": n2}
        results["task2"] = {"summary": s2, "raw_results": all_r2}
        (model_dir / "task2_dynamic_adaptive.json").write_text(json.dumps(results["task2"], ensure_ascii=False, indent=2), encoding="utf-8")
        log(f"    Dynamic+Adaptive: {clean2}/{n2}={s2['clean_rate']:.2f}")
        log(f"    Task 2 done ({time.time()-t_start:.0f}s)")

    elif model_name == "deepseek7b":
        # ============================================================
        # Task 3: DS7B adaptive b (20 prompts)
        # ============================================================
        log(f"  Task 3: DS7B adaptive b (20 prompts)...")
        prompts = EN_PROMPTS_50[:20]
        all_r3 = []
        for pi, prompt in enumerate(prompts):
            input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
            base_gap, ablate_gap = measure_gap(model, input_ids, eos_id, layers, safe_heads, d_head)
            b_adaptive = int(ablate_gap + 2)
            handles = register_multi_head_ablation(layers, safe_heads, d_head)
            h_inj = model.lm_head.register_forward_hook(make_eos_inject_hook(eos_id, b_adaptive, 2))
            try:
                with torch.no_grad():
                    oid = model.generate(input_ids, max_new_tokens=30, do_sample=False, pad_token_id=eos_id)
                gt = oid[0][input_ids.shape[1]:]
                gen = tokenizer.decode(gt, skip_special_tokens=False)
                he = gt[-1].item()==eos_id; ng = len(gt)
            except: gen="ERROR"; he=False; ng=0
            for h in handles: h.remove()
            h_inj.remove()
            ce = evaluate_clean_v4(prompt, gen, he, ng)
            all_r3.append({"prompt": prompt, "base_gap": base_gap, "ablate_gap": ablate_gap,
                           "b": b_adaptive, "generated": gen[:80], "has_eos": he,
                           "n_tokens": ng, "strict_clean": ce["strict_clean"]})
            if (pi+1) % 5 == 0: log(f"    {pi+1}/20 ({time.time()-t_start:.0f}s)")

        n = len(all_r3)
        clean = sum(r["strict_clean"] for r in all_r3)
        eos = sum(r["has_eos"] for r in all_r3)
        s3 = {"clean_rate": clean/n, "eos_rate": eos/n,
              "mean_b": float(np.mean([r["b"] for r in all_r3])), "n": n}
        results["task3"] = {"summary": s3, "raw_results": all_r3}
        (model_dir / "task3_ds7b_adaptive.json").write_text(json.dumps(results["task3"], ensure_ascii=False, indent=2), encoding="utf-8")
        log(f"    DS7B adaptive: {clean}/{n}={s3['clean_rate']:.2f}  EOS: {s3['eos_rate']:.2f}  mean_b={s3['mean_b']:.1f}")
        log(f"    Successes:")
        for r in all_r3:
            if r["strict_clean"]: log(f"      '{r['prompt'][:25]}': b={r['b']} '{r['generated'][:40]}'")
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
    log(f"Tasks: 1=GLM4_adaptive_20p, 2=GLM4_dynamic_adaptive_10p, 3=DS7B_adaptive_20p")
    model_name = sys.argv[1] if len(sys.argv) > 1 else None
    if model_name:
        run_model(model_name)
    else:
        for m in ["glm4", "deepseek7b"]:
            try: run_model(m)
            except Exception as e:
                log(f"  {m} FAILED: {e}"); import traceback; traceback.print_exc()
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()
    log(f"\nPhase {PHASE} complete!")


if __name__ == "__main__":
    main()
