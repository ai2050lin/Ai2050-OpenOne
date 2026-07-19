#!/usr/bin/env python3
"""
Phase 968: 自适应gap控制与动态完成检测闭合审计
================================================
Phase 967发现GLM4 selective hybrid用b=10不够（需b=15）。本阶段：
Task 1: GLM4 selective ablate + b=15 (最关键未测试条件)
Task 2: Adaptive b——per-prompt根据实际gap计算b
Task 3: DS7B大规模hybrid验证(50 prompts)
"""

from __future__ import annotations
import gc, json, sys, time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))
from model_utils import load_model, get_layers, get_model_info, release_model
from phase951_protocol_atlas import ensure_dir
from phase966_natural_stop import (
    log, evaluate_clean_v3, register_multi_head_ablation, get_boundary_ids
)
from phase965_stop_control import DynamicEOSProcessor
from phase964_forward_diff import make_head_hook, make_eos_inject_hook, get_head_dims, EN_PROMPTS_50

PHASE = 968
RESULT_DIR = Path("results/phase968_adaptive_gap")

# Safe heads from Phase 967 (pure_eos_suppressor type)
SAFE_HEADS = {
    "qwen3": [(33, 4), (35, 1), (34, 11), (35, 15), (34, 31)],
    "glm4": [(38, 5), (39, 23), (38, 0), (35, 1)],
    "deepseek7b": [(27, 12), (25, 20), (26, 0), (26, 9), (26, 17), (26, 20), (25, 16), (26, 7), (26, 12), (25, 21)],
}

MODEL_CFG = {
    "qwen3":      {"b_full": 30, "d": 2, "n_adaptive": 10},
    "glm4":       {"b_full": 20, "d": 2, "n_fixed": 10, "n_adaptive": 5},
    "deepseek7b": {"b_full": 20, "d": 2, "n_large": 50},
}


def run_model(model_name):
    log(f"\n{'='*60}\nPhase 968: {model_name}\n{'='*60}")
    model_dir = RESULT_DIR / model_name
    ensure_dir(model_dir)
    cfg = MODEL_CFG[model_name]

    model, tokenizer, device = load_model(model_name)
    info = get_model_info(model, model_name)
    n_heads, d_head = get_head_dims(model, info)
    layers = get_layers(model)
    n_layers = info.n_layers
    eos_id = tokenizer.eos_token_id
    safe_heads = SAFE_HEADS.get(model_name, [])
    boundary_ids = get_boundary_ids(tokenizer)
    log(f"  {info.model_class}, {info.n_layers}L  safe_heads={len(safe_heads)}")

    results = {"model": model_name, "safe_heads": safe_heads}
    t_start = time.time()

    # ============================================================
    # Task 1 (GLM4 only): Selective ablate + b=15
    # ============================================================
    if model_name == "glm4" and "n_fixed" in cfg:
        log(f"  Task 1: GLM4 selective ablate + b=15 ({cfg['n_fixed']} prompts)...")
        prompts = EN_PROMPTS_50[:cfg["n_fixed"]]
        all_r1 = []
        for pi, prompt in enumerate(prompts):
            input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
            # Selective ablate + b=15
            handles = register_multi_head_ablation(layers, safe_heads, d_head)
            h_inj = model.lm_head.register_forward_hook(make_eos_inject_hook(eos_id, 15, cfg["d"]))
            try:
                with torch.no_grad():
                    oid = model.generate(input_ids, max_new_tokens=30, do_sample=False, pad_token_id=eos_id)
                gt = oid[0][input_ids.shape[1]:]
                gen = tokenizer.decode(gt, skip_special_tokens=False)
                he = gt[-1].item()==eos_id; ng = len(gt)
            except Exception as e: gen=f"ERROR:{e}"; he=False; ng=0
            for h in handles: h.remove()
            h_inj.remove()
            ce = evaluate_clean_v3(prompt, gen, he, ng)
            all_r1.append({"prompt": prompt, "generated": gen[:100], "has_eos": he,
                           "n_tokens": ng, "strict_clean": ce["strict_clean"]})
            log(f"    p{pi}: eos={he} clean={ce['strict_clean']} toks={ng} text={gen[:50]}")

        clean_count = sum(r["strict_clean"] for r in all_r1)
        s1 = {"clean_rate": clean_count/len(all_r1), "n": len(all_r1), "b": 15}
        results["task1"] = {"summary": s1, "raw_results": all_r1}
        (model_dir / "task1_glm4_b15.json").write_text(json.dumps(results["task1"], ensure_ascii=False, indent=2), encoding="utf-8")
        log(f"    GLM4 b=15 selective hybrid: {clean_count}/{len(all_r1)} = {s1['clean_rate']:.2f}")
        log(f"    Task 1 done ({time.time()-t_start:.0f}s)")
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    # ============================================================
    # Task 2: Adaptive b (per-prompt gap-based b)
    # ============================================================
    if "n_adaptive" in cfg:
        log(f"  Task 2: Adaptive b ({cfg['n_adaptive']} prompts)...")
        prompts = EN_PROMPTS_50[:cfg["n_adaptive"]]
        all_r2 = []
        for pi, prompt in enumerate(prompts):
            input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
            # Step 1: measure base gap
            with torch.no_grad():
                bl = model(input_ids, use_cache=False).logits[0, -1].detach().float().cpu().numpy()
            base_gap = float(np.sort(bl)[-1]) - (float(bl[eos_id]) if eos_id else 0)
            # Step 2: measure ablated gap
            handles = register_multi_head_ablation(layers, safe_heads, d_head)
            with torch.no_grad():
                al = model(input_ids, use_cache=False).logits[0, -1].detach().float().cpu().numpy()
            for h in handles: h.remove()
            ablate_gap = float(np.sort(al)[-1]) - (float(al[eos_id]) if eos_id else 0)
            # Step 3: adaptive b = ablate_gap + 1
            b_adaptive = int(ablate_gap + 2)  # +2 for safety margin
            # Step 4: generate with ablation + adaptive injection
            handles = register_multi_head_ablation(layers, safe_heads, d_head)
            h_inj = model.lm_head.register_forward_hook(make_eos_inject_hook(eos_id, b_adaptive, cfg["d"]))
            try:
                with torch.no_grad():
                    oid = model.generate(input_ids, max_new_tokens=30, do_sample=False, pad_token_id=eos_id)
                gt = oid[0][input_ids.shape[1]:]
                gen = tokenizer.decode(gt, skip_special_tokens=False)
                he = gt[-1].item()==eos_id; ng = len(gt)
            except Exception as e: gen=f"ERROR:{e}"; he=False; ng=0
            for h in handles: h.remove()
            h_inj.remove()
            ce = evaluate_clean_v3(prompt, gen, he, ng)
            all_r2.append({"prompt": prompt, "base_gap": base_gap, "ablate_gap": ablate_gap,
                           "b_adaptive": b_adaptive, "gap_reduction": base_gap - ablate_gap,
                           "generated": gen[:100], "has_eos": he, "n_tokens": ng,
                           "strict_clean": ce["strict_clean"]})
            log(f"    p{pi}: gap={base_gap:.1f}→{ablate_gap:.1f} b={b_adaptive} eos={he} clean={ce['strict_clean']} text={gen[:40]}")

        clean_count = sum(r["strict_clean"] for r in all_r2)
        s2 = {"clean_rate": clean_count/len(all_r2), "n": len(all_r2),
              "mean_base_gap": float(np.mean([r["base_gap"] for r in all_r2])),
              "mean_ablate_gap": float(np.mean([r["ablate_gap"] for r in all_r2])),
              "mean_b": float(np.mean([r["b_adaptive"] for r in all_r2]))}
        results["task2"] = {"summary": s2, "raw_results": all_r2}
        (model_dir / "task2_adaptive_b.json").write_text(json.dumps(results["task2"], ensure_ascii=False, indent=2), encoding="utf-8")
        log(f"    Adaptive b: {clean_count}/{len(all_r2)} = {s2['clean_rate']:.2f}  (mean_b={s2['mean_b']:.1f})")
        log(f"    Task 2 done ({time.time()-t_start:.0f}s)")
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    # ============================================================
    # Task 3 (DS7B only): Large-scale hybrid validation
    # ============================================================
    if model_name == "deepseek7b" and "n_large" in cfg:
        log(f"  Task 3: DS7B large-scale hybrid ({cfg['n_large']} prompts)...")
        prompts = EN_PROMPTS_50[:cfg["n_large"]]
        all_r3 = []
        for pi, prompt in enumerate(prompts):
            input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
            handles = register_multi_head_ablation(layers, safe_heads, d_head)
            h_inj = model.lm_head.register_forward_hook(make_eos_inject_hook(eos_id, 10, cfg["d"]))
            try:
                with torch.no_grad():
                    oid = model.generate(input_ids, max_new_tokens=30, do_sample=False, pad_token_id=eos_id)
                gt = oid[0][input_ids.shape[1]:]
                gen = tokenizer.decode(gt, skip_special_tokens=False)
                he = gt[-1].item()==eos_id; ng = len(gt)
            except: gen="ERROR"; he=False; ng=0
            for h in handles: h.remove()
            h_inj.remove()
            ce = evaluate_clean_v3(prompt, gen, he, ng)
            all_r3.append({"prompt": prompt, "generated": gen[:80], "has_eos": he,
                           "n_tokens": ng, "strict_clean": ce["strict_clean"]})
            if (pi+1) % 10 == 0: log(f"    {pi+1}/{len(prompts)} ({time.time()-t_start:.0f}s)")

        clean_count = sum(r["strict_clean"] for r in all_r3)
        eos_count = sum(r["has_eos"] for r in all_r3)
        s3 = {"clean_rate": clean_count/len(all_r3), "eos_rate": eos_count/len(all_r3), "n": len(all_r3)}
        results["task3"] = {"summary": s3, "raw_results": all_r3}
        (model_dir / "task3_ds7b_large.json").write_text(json.dumps(results["task3"], ensure_ascii=False, indent=2), encoding="utf-8")
        log(f"    DS7B large-scale: clean={clean_count}/{len(all_r3)}={s3['clean_rate']:.2f}  eos={eos_count}/{len(all_r3)}={s3['eos_rate']:.2f}")
        # Show some successes and failures
        successes = [r for r in all_r3 if r["strict_clean"]][:3]
        failures = [r for r in all_r3 if not r["strict_clean"] and r["has_eos"]][:3]
        log(f"    Successes:")
        for r in successes: log(f"      '{r['prompt'][:25]}': '{r['generated'][:40]}'")
        log(f"    EOS but not clean:")
        for r in failures: log(f"      '{r['prompt'][:25]}': '{r['generated'][:40]}' toks={r['n_tokens']}")
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
    log(f"Tasks: 1=GLM4_b15, 2=adaptive_b, 3=DS7B_large")
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
