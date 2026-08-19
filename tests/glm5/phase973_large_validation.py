#!/usr/bin/env python3
"""
Phase 973: GLM4 50+ prompt大规模验证 + Qwen3高b测试
=====================================================
Task 1: GLM4 65p adaptive+dynamic+MLP channels (验证67.5%稳定性)
Task 2: Qwen3 b=25/30/35测试 (gap>20的解决方案)
Task 3: 三模型gap分布对比 (不同prompt类型)
"""

from __future__ import annotations
import gc, json, sys, time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

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
from phase970_completion_audit import (
    SAFE_HEADS, BoundaryDynamicProcessor, evaluate_clean_v5,
    measure_gap, generate_with_processor, EN_PROMPTS_65,
)
from phase971_protocol_field import make_channel_ablate_hook, get_intermediate_size

PHASE = 973
RESULT_DIR = Path("results/phase973_large_validation")


def task1_glm4_large_scale(model, tokenizer, device, info, layers, eos_id,
                            safe_heads, d_head, boundary_ids, mlp_channels, n_prompts=65):
    """GLM4 65p: adaptive b + dynamic delay + MLP channels + safe heads."""
    log(f"  Task 1: GLM4 {n_prompts}p adaptive+dynamic+MLP channels...")
    prompts = EN_PROMPTS_65[:n_prompts]
    results = []
    t0 = time.time()
    for pi, prompt in enumerate(prompts):
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        # Measure gap with combined ablation
        handles = register_multi_head_ablation(layers, safe_heads, d_head)
        for L, C in mlp_channels:
            h = layers[L].mlp.down_proj.register_forward_pre_hook(make_channel_ablate_hook(C))
            handles.append(h)
        with torch.no_grad():
            al = model(input_ids, use_cache=False).logits[0, -1].float().cpu().numpy()
        for h in handles: h.remove()
        ablate_gap = float(np.sort(al)[-1]) - (float(al[eos_id]) if eos_id else 0)
        b_adaptive = int(ablate_gap + 2)
        # Generate with combined ablation + adaptive b + boundary dynamic
        handles = register_multi_head_ablation(layers, safe_heads, d_head)
        for L, C in mlp_channels:
            h = layers[L].mlp.down_proj.register_forward_pre_hook(make_channel_ablate_hook(C))
            handles.append(h)
        proc = BoundaryDynamicProcessor(eos_id, b_adaptive, min_delay=2, boundary_ids=boundary_ids)
        gen, he, ng = generate_with_processor(model, tokenizer, input_ids, proc, max_new=30, pad_id=eos_id)
        for h in handles: h.remove()
        ce = evaluate_clean_v5(prompt, gen, he, ng)
        results.append({"prompt": prompt, "b": b_adaptive, "ablate_gap": ablate_gap,
                        "generated": gen[:60], "has_eos": he, "n_tokens": ng,
                        "strict_clean": ce["strict_clean"], "has_expected": ce["has_expected"]})
        if (pi+1) % 10 == 0:
            cs = sum(r["strict_clean"] for r in results)
            log(f"    {pi+1}/{n_prompts} clean_so_far={cs}/{pi+1} ({time.time()-t0:.0f}s)")

    n = len(results)
    clean = sum(r["strict_clean"] for r in results)
    eos = sum(r["has_eos"] for r in results)
    expected = sum(r["has_expected"] for r in results)
    mean_b = float(np.mean([r["b"] for r in results]))
    log(f"  RESULT: clean={clean}/{n}={clean/n:.3f}  eos={eos/n:.3f}  "
        f"expected={expected/n:.3f}  mean_b={mean_b:.1f}")
    log(f"  Failures (EOS but not clean):")
    for r in results:
        if r["has_eos"] and not r["strict_clean"]:
            log(f"    '{r['prompt'][:30]}': b={r['b']} '{r['generated'][:45]}' exp={r['has_expected']}")
    log(f"  Task 1 done ({time.time()-t0:.0f}s)")
    return {"summary": {"clean_rate": clean/n, "eos_rate": eos/n, "expected_rate": expected/n,
                         "mean_b": mean_b, "n": n}, "raw_results": results}


def task2_qwen3_high_b(model, tokenizer, device, info, layers, eos_id, boundary_ids):
    """Qwen3: 测试高b值(25/30/35)能否跨过gap>20."""
    log(f"  Task 2: Qwen3 high-b test (b in [20,25,30,35])...")
    biases = [20, 25, 30, 35]
    prompts = EN_PROMPTS_50[:10]
    results = []
    t0 = time.time()
    for b in biases:
        b_clean = 0; b_eos = 0
        for prompt in prompts:
            input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
            proc = BoundaryDynamicProcessor(eos_id, b, min_delay=2, boundary_ids=boundary_ids)
            gen, he, ng = generate_with_processor(model, tokenizer, input_ids, proc, max_new=30, pad_id=eos_id)
            ce = evaluate_clean_v5(prompt, gen, he, ng)
            if ce["strict_clean"]: b_clean += 1
            if he: b_eos += 1
            results.append({"b": b, "prompt": prompt, "generated": gen[:50],
                            "has_eos": he, "clean": ce["strict_clean"]})
        log(f"    b={b:2d}: clean={b_clean}/{len(prompts)}  eos={b_eos}/{len(prompts)}")
    log(f"  Task 2 done ({time.time()-t0:.0f}s)")
    return {"summary": [{"b": b, "clean_rate": sum(1 for r in results if r["b"]==b and r["clean"])/len(prompts),
                          "eos_rate": sum(1 for r in results if r["b"]==b and r["has_eos"])/len(prompts)}
                         for b in biases], "raw_results": results}


def task3_gap_distribution(model, tokenizer, device, eos_id, model_name, n_prompts=50):
    """Task 3: gap分布按prompt类型分析."""
    log(f"  Task 3: Gap distribution analysis ({n_prompts} prompts)...")
    results = []
    t0 = time.time()
    for prompt in EN_PROMPTS_50[:n_prompts]:
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            logits = model(input_ids, use_cache=False).logits[0, -1].float().cpu().numpy()
        top1_val = float(np.sort(logits)[-1])
        top1_id = int(logits.argmax())
        eos_val = float(logits[eos_id]) if eos_id else 0
        gap = top1_val - eos_val
        top1_token = tokenizer.decode([top1_id])
        results.append({"prompt": prompt, "gap": gap, "top1_val": top1_val,
                        "eos_val": eos_val, "top1_token": top1_token})
    gaps = [r["gap"] for r in results]
    log(f"    Gap stats: mean={np.mean(gaps):.2f}  std={np.std(gaps):.2f}  "
        f"min={np.min(gaps):.2f}  max={np.max(gaps):.2f}  median={np.median(gaps):.2f}")
    # Quartiles
    q25, q50, q75 = np.percentile(gaps, [25, 50, 75])
    log(f"    Quartiles: Q25={q25:.2f}  Q50={q50:.2f}  Q75={q75:.2f}")
    # Lowest gap (easiest to stop)
    sorted_by_gap = sorted(results, key=lambda x: x["gap"])
    log(f"    Lowest gap (easiest):")
    for r in sorted_by_gap[:5]:
        log(f"      '{r['prompt'][:30]}': gap={r['gap']:.2f}  top1='{r['top1_token']}'")
    log(f"    Highest gap (hardest):")
    for r in sorted_by_gap[-5:]:
        log(f"      '{r['prompt'][:30]}': gap={r['gap']:.2f}  top1='{r['top1_token']}'")
    log(f"  Task 3 done ({time.time()-t0:.0f}s)")
    return {"mean_gap": float(np.mean(gaps)), "std_gap": float(np.std(gaps)),
            "min_gap": float(np.min(gaps)), "max_gap": float(np.max(gaps)),
            "median_gap": float(np.median(gaps)),
            "quartiles": {"Q25": float(q25), "Q50": float(q50), "Q75": float(q75)},
            "raw_results": results}


def run_glm4():
    """GLM4: Task 1 (65p大规模) + Task 3 (gap分布)."""
    log(f"\n{'='*60}\nPhase 973: GLM4\n{'='*60}")
    model_dir = RESULT_DIR / "glm4"
    ensure_dir(model_dir)
    model, tokenizer, device = load_model("glm4")
    info = get_model_info(model, "glm4")
    n_heads, d_head = get_head_dims(model, info)
    layers = get_layers(model)
    eos_id = tokenizer.eos_token_id
    safe_heads = SAFE_HEADS["glm4"]
    boundary_ids = get_boundary_ids(tokenizer)
    log(f"  {info.model_class}, {info.n_layers}L")

    # Load MLP channels from Phase 971
    try:
        t3_path = Path("results/phase971_protocol_field/glm4/task3_mlp_channel_search.json")
        t3 = json.load(open(t3_path, encoding="utf-8"))
        mlp_channels = []
        for r in t3.get("forward_verified", [])[:6]:
            if r["top1_changed_count"] == 0:
                mlp_channels.append((r["layer"], r["channel"]))
        log(f"  MLP channels from Phase 971: {mlp_channels}")
    except Exception as e:
        log(f"  Could not load MLP channels: {e}, using empty list")
        mlp_channels = []

    t_start = time.time()
    r1 = task1_glm4_large_scale(model, tokenizer, device, info, layers, eos_id,
                                  safe_heads, d_head, boundary_ids, mlp_channels, n_prompts=65)
    (model_dir / "task1_large_scale_65p.json").write_text(
        json.dumps(r1, ensure_ascii=False, indent=2), encoding="utf-8")
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    r3 = task3_gap_distribution(model, tokenizer, device, eos_id, "glm4", n_prompts=50)
    (model_dir / "task3_gap_distribution.json").write_text(
        json.dumps(r3, ensure_ascii=False, indent=2), encoding="utf-8")

    elapsed = time.time() - t_start
    log(f"\n  GLM4 total: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    release_model(model)
    return {"task1": r1, "task3": r3, "elapsed": elapsed}


def run_qwen3():
    """Qwen3: Task 2 (high-b test) + Task 3 (gap分布)."""
    log(f"\n{'='*60}\nPhase 973: Qwen3\n{'='*60}")
    model_dir = RESULT_DIR / "qwen3"
    ensure_dir(model_dir)
    model, tokenizer, device = load_model("qwen3")
    info = get_model_info(model, "qwen3")
    layers = get_layers(model)
    eos_id = tokenizer.eos_token_id
    boundary_ids = get_boundary_ids(tokenizer)
    log(f"  {info.model_class}, {info.n_layers}L")

    t_start = time.time()
    r2 = task2_qwen3_high_b(model, tokenizer, device, info, layers, eos_id, boundary_ids)
    (model_dir / "task2_high_b.json").write_text(
        json.dumps(r2, ensure_ascii=False, indent=2), encoding="utf-8")
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    r3 = task3_gap_distribution(model, tokenizer, device, eos_id, "qwen3", n_prompts=50)
    (model_dir / "task3_gap_distribution.json").write_text(
        json.dumps(r3, ensure_ascii=False, indent=2), encoding="utf-8")

    elapsed = time.time() - t_start
    log(f"\n  Qwen3 total: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    release_model(model)
    return {"task2": r2, "task3": r3, "elapsed": elapsed}


def main():
    ensure_dir(RESULT_DIR)
    log(f"Phase {PHASE} started")
    model_name = sys.argv[1] if len(sys.argv) > 1 else None
    if model_name == "glm4":
        run_glm4()
    elif model_name == "qwen3":
        run_qwen3()
    else:
        # Run both sequentially
        try: run_glm4()
        except Exception as e:
            log(f"  GLM4 FAILED: {e}"); import traceback; traceback.print_exc()
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        try: run_qwen3()
        except Exception as e:
            log(f"  Qwen3 FAILED: {e}"); import traceback; traceback.print_exc()
    log(f"\nPhase {PHASE} complete!")


if __name__ == "__main__":
    main()
