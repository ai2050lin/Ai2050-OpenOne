"""
Phase 303: Cross-Model Comparison — Large-Scale Factorial Decomposition
======================================================================
Compare results across Qwen3, GLM4, DS7B with 60 dual-role tokens each.
Focus on bootstrap stability, cos(R,F) distribution, and per-role-pair patterns.
"""
import sys, os, json, math
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
from pathlib import Path
from collections import defaultdict

RESULT_DIR = Path("results/phase303_large_scale_factorial")
MODEL_NAMES = ["qwen3", "glm4", "deepseek7b"]

def load_results():
    all_data = {}
    for mn in MODEL_NAMES:
        fp = RESULT_DIR / f"{mn}_large_scale_factorial.json"
        if fp.exists():
            with open(fp, "r", encoding="utf-8") as f:
                all_data[mn] = json.load(f)
    return all_data

def cosine_sim(a, b):
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10: return 0.0
    return float(np.dot(a, b) / (na * nb))

def main():
    all_data = load_results()
    print(f"Loaded data for: {list(all_data.keys())}")
    
    # =====================================================================
    # 1. MID-LAYER CAUSAL EFFECTS COMPARISON
    # =====================================================================
    print(f"\n{'='*70}")
    print(f"1. MID-LAYER CAUSAL EFFECTS (R_only, R+F, F_residual)")
    print(f"{'='*70}")
    
    for mn, data in all_data.items():
        nl = data["n_layers"]
        mid_li = str(nl // 2)
        alt_li = str(max(1, nl // 4))
        li = mid_li if mid_li in data["factorial_causal"] else alt_li
        
        layer_res = data["factorial_causal"].get(li, {})
        if not layer_res:
            continue
        
        print(f"\n--- {mn} (Layer {li}) ---")
        
        # Per-role-pair breakdown
        rp_groups = defaultdict(list)
        for key, v in layer_res.items():
            rp = v.get("role_pair", "")
            rp_groups[rp].append(v)
        
        for rp in ["adj_verb", "adj_noun", "noun_verb"]:
            items = rp_groups.get(rp, [])
            if not items: continue
            print(f"\n  {rp} ({len(items)} tokens):")
            
            for metric in ["R_only_cos_shift", "F_only_cos_shift", "R+F_cos_shift", 
                          "R+F_residual_cos_shift", "full_delta_cos_shift", "avg_random_shift"]:
                vals = [v.get(metric) for v in items if v.get(metric) is not None]
                if vals:
                    pos_pct = sum(1 for v in vals if v > 0) / len(vals) * 100
                    print(f"    {metric:30s}: {np.mean(vals):+.4f} ± {np.std(vals):.4f} "
                          f"pos={pos_pct:.0f}% n={len(vals)}")
    
    # =====================================================================
    # 2. COS(R,F) DISTRIBUTION — Factorial vs Residual
    # =====================================================================
    print(f"\n{'='*70}")
    print(f"2. COS(R,F) DISTRIBUTION — Factorial vs Residual")
    print(f"{'='*70}")
    
    for mn, data in all_data.items():
        nl = data["n_layers"]
        mid_li = str(nl // 2)
        alt_li = str(max(1, nl // 4))
        li = mid_li if mid_li in data["factorial_causal"] else alt_li
        
        layer_res = data["factorial_causal"].get(li, {})
        if not layer_res:
            continue
        
        print(f"\n--- {mn} (Layer {li}) ---")
        
        # Overall cos(R,F) distribution
        cos_fact = [v.get("cos_RF_factorial", 0) for v in layer_res.values()]
        cos_resid = [v.get("cos_RF_residual", 0) for v in layer_res.values()]
        
        extreme_fact = sum(1 for v in cos_fact if abs(v) > 0.9)
        extreme_resid = sum(1 for v in cos_resid if abs(v) > 0.9)
        
        print(f"  Factorial cos(R,F): mean={np.mean(cos_fact):+.4f} std={np.std(cos_fact):.4f} "
              f"|cos|>0.9: {extreme_fact}/{len(cos_fact)} ({extreme_fact/len(cos_fact)*100:.0f}%)")
        print(f"  Residual  cos(R,F): mean={np.mean(cos_resid):+.4f} std={np.std(cos_resid):.4f} "
              f"|cos|>0.9: {extreme_resid}/{len(cos_resid)} ({extreme_resid/len(cos_resid)*100:.0f}%)")
        
        # Per-role-pair cos(R,F)
        rp_groups = defaultdict(list)
        for key, v in layer_res.items():
            rp = v.get("role_pair", "")
            rp_groups[rp].append(v)
        
        for rp in ["adj_verb", "adj_noun", "noun_verb"]:
            items = rp_groups.get(rp, [])
            if not items: continue
            cf = [v.get("cos_RF_factorial", 0) for v in items]
            cr = [v.get("cos_RF_residual", 0) for v in items]
            ef = sum(1 for v in cf if abs(v) > 0.9)
            er = sum(1 for v in cr if abs(v) > 0.9)
            print(f"  {rp} factorial: mean={np.mean(cf):+.4f} std={np.std(cf):.4f} |cos|>0.9: {ef}/{len(cf)} ({ef/len(cf)*100:.0f}%)")
            print(f"  {rp} residual:  mean={np.mean(cr):+.4f} std={np.std(cr):.4f} |cos|>0.9: {er}/{len(cr)} ({er/len(cr)*100:.0f}%)")
    
    # =====================================================================
    # 3. DS7B ADJ_NOUN EXTREME COS(R,F) — Detailed Analysis
    # =====================================================================
    print(f"\n{'='*70}")
    print(f"3. DS7B ADJ_NOUN EXTREME COS(R,F) — DETAILED ANALYSIS")
    print(f"{'='*70}")
    
    if "deepseek7b" in all_data:
        data = all_data["deepseek7b"]
        nl = data["n_layers"]
        mid_li = str(nl // 2)
        layer_res = data["factorial_causal"].get(mid_li, {})
        
        if layer_res:
            adj_noun_items = [v for v in layer_res.values() if v.get("role_pair") == "adj_noun"]
            
            print(f"\n  DS7B L{mid_li} adj_noun ({len(adj_noun_items)} tokens):")
            print(f"  {'Token':10s} cos(R,F)_fact cos(R,F)_resid R_only    F_only    R+F_resid R_loo")
            for v in sorted(adj_noun_items, key=lambda x: abs(x.get("cos_RF_factorial", 0)), reverse=True):
                print(f"  {v['token']:10s} {v.get('cos_RF_factorial',0):+8.4f}      "
                      f"{v.get('cos_RF_residual',0):+8.4f}      "
                      f"{v.get('R_only_cos_shift',0):+8.4f}  "
                      f"{v.get('F_only_cos_shift',0):+8.4f}  "
                      f"{v.get('R+F_residual_cos_shift',0):+8.4f}  "
                      f"{v.get('R_loo_cos_shift',0):+8.4f}")
    
    # =====================================================================
    # 4. BOOTSTRAP STABILITY COMPARISON
    # =====================================================================
    print(f"\n{'='*70}")
    print(f"4. BOOTSTRAP STABILITY — KEY METRICS WITH 95% CI")
    print(f"{'='*70}")
    
    for mn, data in all_data.items():
        boot = data.get("bootstrap", {})
        if not boot: continue
        
        print(f"\n--- {mn} ---")
        
        # Show key metrics for "all" group
        for metric in ["R_only_cos_shift", "R+F_cos_shift", "R+F_residual_cos_shift",
                       "full_delta_cos_shift", "avg_random_shift", "cos_RF_factorial"]:
            key = f"all::{metric}"
            if key in boot:
                br = boot[key]
                ci_sig = "***" if (br["ci_low"] > 0 or br["ci_high"] < 0) else ""
                print(f"  {metric:30s}: {br['mean']:+.4f} CI=[{br['ci_low']:+.4f}, {br['ci_high']:+.4f}] "
                      f"width={br['ci_width']:.4f} pos={br['positive_pct']:.0f}% {ci_sig}")
        
        # Per-role-pair R_only
        print(f"\n  Per-role-pair R_only:")
        for rp in ["adj_verb", "adj_noun", "noun_verb"]:
            key = f"{rp}::R_only_cos_shift"
            if key in boot:
                br = boot[key]
                ci_sig = "***" if (br["ci_low"] > 0 or br["ci_high"] < 0) else ""
                print(f"    {rp:12s}: {br['mean']:+.4f} CI=[{br['ci_low']:+.4f}, {br['ci_high']:+.4f}] "
                      f"pos={br['positive_pct']:.0f}% {ci_sig}")
    
    # =====================================================================
    # 5. R_loo GENERALIZATION — Cross-Model Comparison
    # =====================================================================
    print(f"\n{'='*70}")
    print(f"5. R_loo GENERALIZATION — Cross-Model Comparison")
    print(f"{'='*70}")
    
    for mn, data in all_data.items():
        nl = data["n_layers"]
        mid_li = str(nl // 2)
        layer_res = data["factorial_causal"].get(mid_li, {})
        if not layer_res: continue
        
        rp_groups = defaultdict(list)
        for key, v in layer_res.items():
            rp = v.get("role_pair", "")
            rp_groups[rp].append(v)
        
        print(f"\n  {mn}:")
        for rp in ["adj_verb", "adj_noun", "noun_verb", "all"]:
            if rp == "all":
                items = list(layer_res.values())
            else:
                items = rp_groups.get(rp, [])
            if not items: continue
            
            R_only = [v.get("R_only_cos_shift", 0) for v in items if v.get("R_only_cos_shift") is not None]
            R_loo = [v.get("R_loo_cos_shift", 0) for v in items if v.get("R_loo_cos_shift") is not None]
            if R_only and R_loo:
                ratio = np.mean(R_loo) / max(np.mean(R_only), 1e-6)
                print(f"    {rp:12s}: R_loo/R_only = {ratio:.3f} (R_loo={np.mean(R_loo):+.4f}, R_only={np.mean(R_only):+.4f})")
    
    # =====================================================================
    # 6. FULL-DELTA RECONSTRUCTION — How much of full_delta causal effect 
    #    is captured by R_only, R+F, R+F_residual?
    # =====================================================================
    print(f"\n{'='*70}")
    print(f"6. FULL-DELTA RECONSTRUCTION QUALITY")
    print(f"{'='*70}")
    
    for mn, data in all_data.items():
        nl = data["n_layers"]
        mid_li = str(nl // 2)
        layer_res = data["factorial_causal"].get(mid_li, {})
        if not layer_res: continue
        
        print(f"\n  {mn} (L{mid_li}):")
        
        full_cs = [v.get("full_delta_cos_shift", 0) for v in layer_res.values() 
                   if v.get("full_delta_cos_shift") is not None]
        R_cs = [v.get("R_only_cos_shift", 0) for v in layer_res.values()
                if v.get("R_only_cos_shift") is not None]
        RpF_cs = [v.get("R+F_cos_shift", 0) for v in layer_res.values()
                  if v.get("R+F_cos_shift") is not None]
        RpFr_cs = [v.get("R+F_residual_cos_shift", 0) for v in layer_res.values()
                   if v.get("R+F_residual_cos_shift") is not None]
        
        if full_cs and R_cs:
            R_pct = np.mean(R_cs) / max(np.mean(full_cs), 1e-6) * 100
            print(f"    R_only / full_delta = {R_pct:.1f}%")
        if full_cs and RpF_cs:
            RpF_pct = np.mean(RpF_cs) / max(np.mean(full_cs), 1e-6) * 100
            print(f"    R+F / full_delta = {RpF_pct:.1f}%")
        if full_cs and RpFr_cs:
            RpFr_pct = np.mean(RpFr_cs) / max(np.mean(full_cs), 1e-6) * 100
            print(f"    R+F_residual / full_delta = {RpFr_pct:.1f}%")
    
    # =====================================================================
    # 7. LAYER PROGRESSION — R_only across layers
    # =====================================================================
    print(f"\n{'='*70}")
    print(f"7. R_only LAYER PROGRESSION")
    print(f"{'='*70}")
    
    for mn, data in all_data.items():
        print(f"\n  {mn}:")
        for li_str in sorted(data["factorial_causal"].keys(), key=int):
            layer_res = data["factorial_causal"][li_str]
            R_cs = [v.get("R_only_cos_shift", 0) for v in layer_res.values()
                    if v.get("R_only_cos_shift") is not None]
            RpFr_cs = [v.get("R+F_residual_cos_shift", 0) for v in layer_res.values()
                       if v.get("R+F_residual_cos_shift") is not None]
            if R_cs:
                print(f"    L{li_str:>3s}: R_only={np.mean(R_cs):+.4f} "
                      f"R+F_resid={np.mean(RpFr_cs):+.4f}" if RpFr_cs else 
                      f"    L{li_str:>3s}: R_only={np.mean(R_cs):+.4f}")
    
    # =====================================================================
    # 8. SUMMARY TABLE — Key metrics across models at mid layer
    # =====================================================================
    print(f"\n{'='*70}")
    print(f"8. SUMMARY TABLE — Key Metrics at Mid Layer")
    print(f"{'='*70}")
    
    print(f"\n  {'Model':12s} {'R_only':>8s} {'R+F':>8s} {'R+F_res':>8s} {'full_d':>8s} "
          f"{'random':>8s} {'|cos|>0.9':>10s} {'R_loo/R':>8s}")
    
    for mn, data in all_data.items():
        nl = data["n_layers"]
        mid_li = str(nl // 2)
        layer_res = data["factorial_causal"].get(mid_li, {})
        if not layer_res: continue
        
        R_cs = [v.get("R_only_cos_shift", 0) for v in layer_res.values() if v.get("R_only_cos_shift") is not None]
        RpF_cs = [v.get("R+F_cos_shift", 0) for v in layer_res.values() if v.get("R+F_cos_shift") is not None]
        RpFr_cs = [v.get("R+F_residual_cos_shift", 0) for v in layer_res.values() if v.get("R+F_residual_cos_shift") is not None]
        full_cs = [v.get("full_delta_cos_shift", 0) for v in layer_res.values() if v.get("full_delta_cos_shift") is not None]
        rand_cs = [v.get("avg_random_shift", 0) for v in layer_res.values()]
        cos_fact = [abs(v.get("cos_RF_factorial", 0)) for v in layer_res.values()]
        R_loo_cs = [v.get("R_loo_cos_shift", 0) for v in layer_res.values() if v.get("R_loo_cos_shift") is not None]
        
        extreme_pct = sum(1 for v in cos_fact if v > 0.9) / len(cos_fact) * 100 if cos_fact else 0
        r_loo_ratio = np.mean(R_loo_cs) / max(np.mean(R_cs), 1e-6) if R_loo_cs and R_cs else 0
        
        print(f"  {mn:12s} {np.mean(R_cs):+8.4f} {np.mean(RpF_cs):+8.4f} "
              f"{np.mean(RpFr_cs):+8.4f} {np.mean(full_cs):+8.4f} "
              f"{np.mean(rand_cs):+8.4f} {extreme_pct:8.1f}% {r_loo_ratio:8.3f}")
    
    print(f"\n  Note: F_only and R+F+RF are ~0 for all models due to unbalanced design artifact.")
    print(f"  F_direction_avg ≈ 0 when frames are role-specific (no shared frames across roles).")
    print(f"  RF_direction = -R_direction by construction in unbalanced design.")


if __name__ == "__main__":
    main()
