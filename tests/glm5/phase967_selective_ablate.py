#!/usr/bin/env python3
"""
Phase 967: Selective ablate——根据head功能分类选择性消融
======================================================
Phase 966发现：ablate所有gap-widening heads会改变内容(qwen3)或触发切换(GLM4)。
本阶段：分类heads，只ablate"纯EOS抑制head"(ΔEOS>0.1, |Δtop1|<0.15)，保留模式锁定head。

Task 1: 全层Δgap搜索 + head功能分类
Task 2: Selective ablate(只ablate safe heads) + gap测量
Task 3: Selective ablate + hybrid(减小注入)
Task 4: DS7B大规模hybrid验证(20 prompts)
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
    log, evaluate_clean_v3, register_multi_head_ablation, SEMANTIC_EQUIV, SPECIAL_TOKENS
)
from phase965_stop_control import DynamicEOSProcessor, get_boundary_ids
from phase964_forward_diff import make_head_hook, make_eos_inject_hook, get_head_dims, EN_PROMPTS_50

PHASE = 967
RESULT_DIR = Path("results/phase967_selective_ablate")

MODEL_CFG = {
    "qwen3":      {"b": 30, "d": 2, "n_search": 3, "n_test": 10, "n_val": 20, "search_L": 5},
    "glm4":       {"b": 20, "d": 2, "n_search": 2, "n_test": 3,  "n_val": 10, "search_L": 5},
    "deepseek7b": {"b": 20, "d": 2, "n_search": 3, "n_test": 5,  "n_val": 20, "search_L": 3},
}


def classify_head(delta_gap, delta_top1, delta_eos):
    """Classify head function based on ablation effects."""
    if delta_gap >= 0:
        return "neutral"  # ablation doesn't narrow gap
    if delta_eos > 0.1 and abs(delta_top1) < 0.15:
        return "pure_eos_suppressor"  # SAFE: ablate increases EOS without changing top1
    if delta_eos < -0.1 and delta_top1 < -0.5:
        return "top1_promoter"  # SAFE: ablate decreases top1 without changing EOS much
    if delta_eos > 0.1 and delta_top1 < -0.15:
        return "mode_lock_or_content"  # RISKY: ablate changes both
    if abs(delta_top1) > 0.3:
        return "content_generator"  # RISKY: ablate changes content
    return "weak_effect"  # small effect, safe but not useful


def run_model(model_name):
    log(f"\n{'='*60}\nPhase 967: {model_name}\n{'='*60}")
    model_dir = RESULT_DIR / model_name
    ensure_dir(model_dir)
    cfg = MODEL_CFG[model_name]

    model, tokenizer, device = load_model(model_name)
    info = get_model_info(model, model_name)
    n_heads, d_head = get_head_dims(model, info)
    layers = get_layers(model)
    n_layers = info.n_layers
    eos_id = tokenizer.eos_token_id
    boundary_ids = get_boundary_ids(tokenizer)
    log(f"  {info.model_class}, {info.n_layers}L")

    results = {"model": model_name}
    t_start = time.time()

    # ============================================================
    # Task 1: Δgap search + head classification
    # ============================================================
    log(f"  Task 1: Δgap search + classification ({cfg['search_L']} layers, {cfg['n_search']} prompts)...")
    search_layers = list(range(max(0, n_layers - cfg["search_L"]), n_layers))
    gap_data = defaultdict(lambda: {"delta_gap": [], "delta_top1": [], "delta_eos": []})

    for pi, prompt in enumerate(EN_PROMPTS_50[:cfg["n_search"]]):
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            bl = model(input_ids, use_cache=False).logits[0, -1].detach().float().cpu().numpy()
        base_top1 = float(np.sort(bl)[-1]); base_eos = float(bl[eos_id]) if eos_id else 0
        base_gap = base_top1 - base_eos

        for L in search_layers:
            for H in range(n_heads):
                sc = H * d_head; ec = sc + d_head
                handle = layers[L].self_attn.o_proj.register_forward_pre_hook(make_head_hook(sc, ec, 0.0))
                try:
                    with torch.no_grad():
                        pl = model(input_ids, use_cache=False).logits[0, -1].detach().float().cpu().numpy()
                except: pl = bl.copy()
                handle.remove()
                p_top1 = float(np.sort(pl)[-1]); p_eos = float(pl[eos_id]) if eos_id else 0
                key = f"L{L}_H{H}"
                gap_data[key]["delta_gap"].append((p_top1 - p_eos) - base_gap)
                gap_data[key]["delta_top1"].append(p_top1 - base_top1)
                gap_data[key]["delta_eos"].append(p_eos - base_eos)
        log(f"    {pi+1}/{cfg['n_search']} ({time.time()-t_start:.0f}s)")

    # Aggregate + classify
    head_info = {}
    for key, d in gap_data.items():
        dg = float(np.mean(d["delta_gap"]))
        dt = float(np.mean(d["delta_top1"]))
        de = float(np.mean(d["delta_eos"]))
        head_info[key] = {"delta_gap": dg, "delta_top1": dt, "delta_eos": de,
                           "classification": classify_head(dg, dt, de)}

    # Select safe heads
    safe_heads = [(k, v) for k, v in head_info.items() if v["classification"] in ("pure_eos_suppressor", "top1_promoter")]
    safe_heads.sort(key=lambda x: x[1]["delta_gap"])  # Most gap-narrowing first

    # Also get all gap-narrowing heads for comparison
    all_narrowing = [(k, v) for k, v in head_info.items() if v["delta_gap"] < -0.01]
    all_narrowing.sort(key=lambda x: x[1]["delta_gap"])

    results["task1"] = {
        "search_layers": search_layers, "base_gap": base_gap,
        "safe_heads": [(k, v["delta_gap"], v["delta_top1"], v["delta_eos"], v["classification"]) for k, v in safe_heads[:20]],
        "all_narrowing_top10": [(k, v["delta_gap"], v["delta_top1"], v["delta_eos"], v["classification"]) for k, v in all_narrowing[:10]],
        "classification_counts": dict(defaultdict(int, **{c: sum(1 for v in head_info.values() if v["classification"] == c) for c in set(v["classification"] for v in head_info.values())})),
    }
    log(f"    Base gap: {base_gap:.3f}")
    log(f"    Classification counts: {results['task1']['classification_counts']}")
    log(f"    Safe heads (pure_eos_suppressor + top1_promoter): {len(safe_heads)}")
    for k, v in safe_heads[:5]:
        log(f"      {k}: Δgap={v['delta_gap']:.4f} Δtop1={v['delta_top1']:.4f} ΔEOS={v['delta_eos']:.4f} [{v['classification']}]")
    log(f"    All narrowing top 5 (for comparison):")
    for k, v in all_narrowing[:5]:
        log(f"      {k}: Δgap={v['delta_gap']:.4f} Δtop1={v['delta_top1']:.4f} ΔEOS={v['delta_eos']:.4f} [{v['classification']}]")
    log(f"    Task 1 done ({time.time()-t_start:.0f}s)")

    if torch.cuda.is_available(): torch.cuda.empty_cache()

    # ============================================================
    # Task 2+3: Selective ablate + hybrid
    # ============================================================
    log(f"  Task 2+3: Selective ablate + hybrid ({cfg['n_test']} prompts)...")

    # Parse safe heads
    safe_head_specs = []
    for k, _ in safe_heads[:10]:  # Top 10 safe heads
        parts = k.split("_")
        safe_head_specs.append((int(parts[0][1:]), int(parts[1][1:])))

    # Also parse all narrowing heads (for comparison)
    all_head_specs = []
    for k, _ in all_narrowing[:10]:
        parts = k.split("_")
        all_head_specs.append((int(parts[0][1:]), int(parts[1][1:])))

    log(f"    Safe heads to ablate: {safe_head_specs[:5]}...")
    log(f"    All heads (comparison): {all_head_specs[:5]}...")

    prompts_test = EN_PROMPTS_50[:cfg["n_test"]]
    test_results = []

    for pi, prompt in enumerate(prompts_test):
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

        # Condition 1: Normal
        with torch.no_grad():
            oid = model.generate(input_ids, max_new_tokens=30, do_sample=False, pad_token_id=eos_id)
        gt = oid[0][input_ids.shape[1]:]
        gen_n = tokenizer.decode(gt, skip_special_tokens=False)
        ce_n = evaluate_clean_v3(prompt, gen_n, gt[-1].item()==eos_id, len(gt))

        # Condition 2: Selective ablate only (safe heads)
        handles = register_multi_head_ablation(layers, safe_head_specs, d_head)
        with torch.no_grad():
            oid = model.generate(input_ids, max_new_tokens=30, do_sample=False, pad_token_id=eos_id)
        gt = oid[0][input_ids.shape[1]:]
        gen_sa = tokenizer.decode(gt, skip_special_tokens=False)
        ce_sa = evaluate_clean_v3(prompt, gen_sa, gt[-1].item()==eos_id, len(gt))
        for h in handles: h.remove()

        # Condition 3: Selective ablate + reduced injection (hybrid)
        handles = register_multi_head_ablation(layers, safe_head_specs, d_head)
        h_inj = model.lm_head.register_forward_hook(make_eos_inject_hook(eos_id, cfg["b"]//2, cfg["d"]))
        with torch.no_grad():
            oid = model.generate(input_ids, max_new_tokens=30, do_sample=False, pad_token_id=eos_id)
        gt = oid[0][input_ids.shape[1]:]
        gen_h = tokenizer.decode(gt, skip_special_tokens=False)
        ce_h = evaluate_clean_v3(prompt, gen_h, gt[-1].item()==eos_id, len(gt))
        for h in handles: h.remove()
        h_inj.remove()

        # Condition 4: All ablate + reduced injection (Phase 966 style, for comparison)
        handles = register_multi_head_ablation(layers, all_head_specs, d_head)
        h_inj = model.lm_head.register_forward_hook(make_eos_inject_hook(eos_id, cfg["b"]//2, cfg["d"]))
        with torch.no_grad():
            oid = model.generate(input_ids, max_new_tokens=30, do_sample=False, pad_token_id=eos_id)
        gt = oid[0][input_ids.shape[1]:]
        gen_ah = tokenizer.decode(gt, skip_special_tokens=False)
        ce_ah = evaluate_clean_v3(prompt, gen_ah, gt[-1].item()==eos_id, len(gt))
        for h in handles: h.remove()
        h_inj.remove()

        test_results.append({
            "prompt": prompt,
            "normal_clean": ce_n["strict_clean"], "normal_gen": gen_n[:80],
            "selective_ablate_clean": ce_sa["strict_clean"], "sa_gen": gen_sa[:80],
            "selective_hybrid_clean": ce_h["strict_clean"], "hybrid_gen": gen_h[:80],
            "all_hybrid_clean": ce_ah["strict_clean"], "all_hybrid_gen": gen_ah[:80],
        })
        if (pi+1) % 3 == 0: log(f"    {pi+1}/{len(prompts_test)} ({time.time()-t_start:.0f}s)")

    n = len(test_results)
    s23 = {
        "normal_clean_rate": sum(r["normal_clean"] for r in test_results)/n,
        "selective_ablate_clean_rate": sum(r["selective_ablate_clean"] for r in test_results)/n,
        "selective_hybrid_clean_rate": sum(r["selective_hybrid_clean"] for r in test_results)/n,
        "all_hybrid_clean_rate": sum(r["all_hybrid_clean"] for r in test_results)/n,
    }
    results["task2_3"] = {"summary": s23, "safe_heads": safe_head_specs, "all_heads": all_head_specs,
                           "raw_results": test_results}
    (model_dir / "task2_3_selective.json").write_text(json.dumps(results["task2_3"], ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"    Results:")
    log(f"      Normal:              {s23['normal_clean_rate']:.2f}")
    log(f"      Selective ablate:    {s23['selective_ablate_clean_rate']:.2f}")
    log(f"      Selective hybrid:    {s23['selective_hybrid_clean_rate']:.2f}")
    log(f"      All hybrid(Phase966):{s23['all_hybrid_clean_rate']:.2f}")
    log(f"    Samples (p0):")
    r = test_results[0]
    log(f"      normal:    '{r['normal_gen'][:50]}'")
    log(f"      sel_ablate:'{r['sa_gen'][:50]}'")
    log(f"      sel_hybrid:'{r['hybrid_gen'][:50]}'")
    log(f"      all_hybrid:'{r['all_hybrid_gen'][:50]}'")
    log(f"    Task 2+3 done ({time.time()-t_start:.0f}s)")

    if torch.cuda.is_available(): torch.cuda.empty_cache()

    # ============================================================
    # Task 4: Large-scale validation for best condition
    # ============================================================
    # Determine best condition
    best_clean = max(s23["selective_hybrid_clean_rate"], s23["all_hybrid_clean_rate"])
    if s23["selective_hybrid_clean_rate"] >= s23["all_hybrid_clean_rate"]:
        best_heads = safe_head_specs
        best_name = "selective_hybrid"
    else:
        best_heads = all_head_specs
        best_name = "all_hybrid"

    log(f"  Task 4: Large-scale validation ({best_name}, {cfg['n_val']} prompts)...")
    prompts_val = EN_PROMPTS_50[:cfg["n_val"]]
    val_results = []
    for pi, prompt in enumerate(prompts_val):
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        handles = register_multi_head_ablation(layers, best_heads, d_head)
        h_inj = model.lm_head.register_forward_hook(make_eos_inject_hook(eos_id, cfg["b"]//2, cfg["d"]))
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
        val_results.append({"prompt": prompt, "generated": gen[:80], "has_eos": he,
                             "n_tokens": ng, "strict_clean": ce["strict_clean"]})
        if (pi+1) % 10 == 0: log(f"    {pi+1}/{len(prompts_val)} ({time.time()-t_start:.0f}s)")

    val_clean = sum(r["strict_clean"] for r in val_results)
    s4 = {"method": best_name, "clean_rate": val_clean/len(val_results), "n": len(val_results)}
    results["task4"] = {"summary": s4, "raw_results": val_results}
    (model_dir / "task4_validation.json").write_text(json.dumps(results["task4"], ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"    Large-scale {best_name}: {val_clean}/{len(val_results)} = {s4['clean_rate']:.2f}")
    for r in val_results[:3]:
        log(f"      '{r['prompt'][:25]}': clean={r['strict_clean']} text='{r['generated'][:40]}'")
    log(f"    Task 4 done ({time.time()-t_start:.0f}s)")

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
    log(f"Tasks: 1=gap_search+classify, 2-3=selective_ablate+hybrid, 4=large_scale")
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
