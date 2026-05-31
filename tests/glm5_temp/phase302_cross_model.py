"""
Phase 302 Cross-Model Analysis
================================
Compare factorial vs residual F decomposition across Qwen3, GLM4, DS7B.
Key questions:
1. Does factorial F differ from residual F? (Is Phase 301's F definition problematic?)
2. Is RF binding term real? (R+F+RF vs R+F)
3. Does the DS7B extreme cos(R,F) pattern persist with factorial F?
4. Per-role-pair breakdown
"""
import sys, json, numpy as np
from pathlib import Path
from collections import defaultdict

sys.stdout.reconfigure(encoding='utf-8')

RESULT_DIR = Path("results/phase302_factorial_decomposition")

def load_results(model_name):
    path = RESULT_DIR / f"{model_name}_factorial_decomposition.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def main():
    models = ["qwen3", "glm4", "deepseek7b"]
    data = {m: load_results(m) for m in models}
    
    print("=" * 80)
    print("PHASE 302: CROSS-MODEL FACTORIAL DECOMPOSITION ANALYSIS")
    print("=" * 80)
    
    # =====================================================================
    # 1. F_factorial vs F_residual: Are they the same?
    # =====================================================================
    print("\n" + "=" * 80)
    print("1. F_FACTORIAL vs F_RESIDUAL: Is Phase 301's F definition reliable?")
    print("=" * 80)
    
    for model_name in models:
        d = data[model_name]
        nl = d["n_layers"]
        mid_li = str(nl // 2)
        
        print(f"\n--- {model_name} ---")
        for li_str in sorted(d["factorial_causal"].keys(), key=lambda x: int(x)):
            layer_res = d["factorial_causal"][li_str]
            if not layer_res: continue
            
            cos_F_vals = [v.get("cos_F_fact_resid", 0) for v in layer_res.values()
                         if v.get("cos_F_fact_resid") is not None]
            cos_RF_fact = [v.get("cos_RF_factorial", 0) for v in layer_res.values()
                          if v.get("cos_RF_factorial") is not None]
            cos_RF_resid = [v.get("cos_RF_residual", 0) for v in layer_res.values()
                           if v.get("cos_RF_residual") is not None]
            
            if cos_F_vals:
                print(f"  L{li_str}: cos(F_fact, F_resid) = {np.mean(cos_F_vals):+.4f} | "
                      f"cos(R,F)_fact = {np.mean(cos_RF_fact):+.4f} | "
                      f"cos(R,F)_resid = {np.mean(cos_RF_resid):+.4f}")
    
    # =====================================================================
    # 2. Core: R+F vs R+F+RF — Is the binding term real?
    # =====================================================================
    print("\n" + "=" * 80)
    print("2. R+F vs R+F+RF: IS THE RF BINDING TERM REAL?")
    print("=" * 80)
    
    for model_name in models:
        d = data[model_name]
        nl = d["n_layers"]
        print(f"\n--- {model_name} ---")
        
        for li_str in sorted(d["factorial_causal"].keys(), key=lambda x: int(x)):
            layer_res = d["factorial_causal"][li_str]
            if not layer_res: continue
            
            R_only = [v.get("R_only_cos_shift") for v in layer_res.values() if v.get("R_only_cos_shift") is not None]
            F_only = [v.get("F_only_cos_shift") for v in layer_res.values() if v.get("F_only_cos_shift") is not None]
            RF_only = [v.get("RF_only_cos_shift") for v in layer_res.values() if v.get("RF_only_cos_shift") is not None]
            RpF = [v.get("R+F_cos_shift") for v in layer_res.values() if v.get("R+F_cos_shift") is not None]
            RpFpRF = [v.get("R+F+RF_cos_shift") for v in layer_res.values() if v.get("R+F+RF_cos_shift") is not None]
            full = [v.get("full_delta_cos_shift") for v in layer_res.values() if v.get("full_delta_cos_shift") is not None]
            rand = [v.get("avg_random_shift", 0) for v in layer_res.values()]
            RpF_resid = [v.get("R+F_residual_cos_shift") for v in layer_res.values() if v.get("R+F_residual_cos_shift") is not None]
            
            rf_boost = np.mean(RpFpRF) - np.mean(RpF) if RpF and RpFpRF else 0
            
            print(f"  L{li_str}: R={np.mean(R_only):+.4f} F={np.mean(F_only):+.4f} RF={np.mean(RF_only):+.4f} | "
                  f"R+F={np.mean(RpF):+.4f} R+F+RF={np.mean(RpFpRF):+.4f} | "
                  f"full={np.mean(full):+.4f} R+F_resid={np.mean(RpF_resid):+.4f} | "
                  f"RF_boost={rf_boost:+.4f} random={np.mean(rand):+.4f}")
    
    # =====================================================================
    # 3. DS7B cos(R,F) distribution: factorial vs residual
    # =====================================================================
    print("\n" + "=" * 80)
    print("3. DS7B cos(R,F) EXTREME PATTERN: FACTORIAL vs RESIDUAL")
    print("=" * 80)
    
    for model_name in models:
        d = data[model_name]
        print(f"\n--- {model_name} ---")
        
        all_cos_fact = []; all_cos_resid = []
        for li_str, layer_res in d["factorial_causal"].items():
            for key, v in layer_res.items():
                cv = v.get("cos_RF_factorial")
                if cv is not None: all_cos_fact.append(cv)
                cv2 = v.get("cos_RF_residual")
                if cv2 is not None: all_cos_resid.append(cv2)
        
        if all_cos_fact:
            extreme_fact = sum(1 for v in all_cos_fact if abs(v) > 0.9)
            print(f"  Factorial: mean={np.mean(all_cos_fact):+.4f}, std={np.std(all_cos_fact):.4f}, "
                  f"|cos|>0.9: {extreme_fact}/{len(all_cos_fact)} ({100*extreme_fact/len(all_cos_fact):.1f}%)")
        if all_cos_resid:
            extreme_resid = sum(1 for v in all_cos_resid if abs(v) > 0.9)
            print(f"  Residual:  mean={np.mean(all_cos_resid):+.4f}, std={np.std(all_cos_resid):.4f}, "
                  f"|cos|>0.9: {extreme_resid}/{len(all_cos_resid)} ({100*extreme_resid/len(all_cos_resid):.1f}%)")
    
    # =====================================================================
    # 4. Per-role-pair breakdown at mid-layer
    # =====================================================================
    print("\n" + "=" * 80)
    print("4. PER-ROLE-PAIR BREAKDOWN AT MID-LAYER")
    print("=" * 80)
    
    for model_name in models:
        d = data[model_name]
        nl = d["n_layers"]
        mid_li = str(nl // 2)
        layer_res = d["factorial_causal"].get(mid_li, {})
        if not layer_res: continue
        
        print(f"\n--- {model_name} (L{mid_li}) ---")
        
        rp_groups = defaultdict(list)
        for key, v in layer_res.items():
            rp = v.get("role_pair", "")
            rp_groups[rp].append(v)
        
        for rp in ["adj_verb", "adj_noun", "noun_verb"]:
            items = rp_groups.get(rp, [])
            if not items: continue
            
            R = [v.get("R_only_cos_shift", 0) for v in items]
            F = [v.get("F_only_cos_shift", 0) for v in items]
            RF = [v.get("RF_only_cos_shift", 0) for v in items]
            RpF = [v.get("R+F_cos_shift", 0) for v in items]
            RpFpRF = [v.get("R+F+RF_cos_shift", 0) for v in items]
            RpF_resid = [v.get("R+F_residual_cos_shift", 0) for v in items]
            
            print(f"  {rp} ({len(items)} tokens):")
            print(f"    R_only={np.mean(R):+.4f} F_only={np.mean(F):+.4f} RF_only={np.mean(RF):+.4f}")
            print(f"    R+F={np.mean(RpF):+.4f} R+F+RF={np.mean(RpFpRF):+.4f} R+F_resid={np.mean(RpF_resid):+.4f}")
            print(f"    RF boost = {np.mean(RpFpRF) - np.mean(RpF):+.4f}")
    
    # =====================================================================
    # 5. Deep layer analysis (critical for DS7B)
    # =====================================================================
    print("\n" + "=" * 80)
    print("5. DEEP LAYER ANALYSIS")
    print("=" * 80)
    
    for model_name in models:
        d = data[model_name]
        nl = d["n_layers"]
        deep_li = str(nl - 2)
        layer_res = d["factorial_causal"].get(deep_li, {})
        if not layer_res: continue
        
        R = [v.get("R_only_cos_shift") for v in layer_res.values() if v.get("R_only_cos_shift") is not None]
        F = [v.get("F_only_cos_shift") for v in layer_res.values() if v.get("F_only_cos_shift") is not None]
        RF = [v.get("RF_only_cos_shift") for v in layer_res.values() if v.get("RF_only_cos_shift") is not None]
        RpF = [v.get("R+F_cos_shift") for v in layer_res.values() if v.get("R+F_cos_shift") is not None]
        RpFpRF = [v.get("R+F+RF_cos_shift") for v in layer_res.values() if v.get("R+F+RF_cos_shift") is not None]
        full = [v.get("full_delta_cos_shift") for v in layer_res.values() if v.get("full_delta_cos_shift") is not None]
        
        print(f"\n--- {model_name} (L{deep_li}, deep) ---")
        if R: print(f"  R_only={np.mean(R):+.4f} pos={sum(1 for s in R if s>0)}/{len(R)}")
        if F: print(f"  F_only={np.mean(F):+.4f} pos={sum(1 for s in F if s>0)}/{len(F)}")
        if RF: print(f"  RF_only={np.mean(RF):+.4f} pos={sum(1 for s in RF if s>0)}/{len(RF)}")
        if RpF: print(f"  R+F={np.mean(RpF):+.4f}")
        if RpFpRF: print(f"  R+F+RF={np.mean(RpFpRF):+.4f}")
        if full: print(f"  full_delta={np.mean(full):+.4f}")
        if RpF and RpFpRF:
            print(f"  RF boost = {np.mean(RpFpRF) - np.mean(RpF):+.4f}")
    
    # =====================================================================
    # 6. F_only causal effect across layers
    # =====================================================================
    print("\n" + "=" * 80)
    print("6. F_ONLY CAUSAL EFFECT: IS FACTORIAL F A REAL COMPONENT?")
    print("=" * 80)
    
    for model_name in models:
        d = data[model_name]
        print(f"\n--- {model_name} ---")
        
        for li_str in sorted(d["factorial_causal"].keys(), key=lambda x: int(x)):
            layer_res = d["factorial_causal"][li_str]
            if not layer_res: continue
            
            F_only = [v.get("F_only_cos_shift") for v in layer_res.values() if v.get("F_only_cos_shift") is not None]
            F_resid = [v.get("F_residual_cos_shift") for v in layer_res.values() if v.get("F_residual_cos_shift") is not None]
            rand = [v.get("avg_random_shift", 0) for v in layer_res.values()]
            
            f_fact_vs_rand = np.mean(F_only) / max(abs(np.mean(rand)), 0.001) if F_only else 0
            f_resid_vs_rand = np.mean(F_resid) / max(abs(np.mean(rand)), 0.001) if F_resid else 0
            
            print(f"  L{li_str}: F_fact={np.mean(F_only):+.4f} ({f_fact_vs_rand:.1f}x rand) | "
                  f"F_resid={np.mean(F_resid):+.4f} ({f_resid_vs_rand:.1f}x rand) | "
                  f"rand={np.mean(rand):+.4f}")
    
    # =====================================================================
    # 7. R_loo generalization
    # =====================================================================
    print("\n" + "=" * 80)
    print("7. R_LOO GENERALIZATION: IS R SHARED ACROSS TOKENS?")
    print("=" * 80)
    
    for model_name in models:
        d = data[model_name]
        nl = d["n_layers"]
        mid_li = str(nl // 2)
        layer_res = d["factorial_causal"].get(mid_li, {})
        if not layer_res: continue
        
        R_only = [v.get("R_only_cos_shift") for v in layer_res.values() if v.get("R_only_cos_shift") is not None]
        R_loo = [v.get("R_loo_cos_shift") for v in layer_res.values() if v.get("R_loo_cos_shift") is not None]
        
        print(f"\n--- {model_name} (L{mid_li}) ---")
        if R_only: print(f"  R_only={np.mean(R_only):+.4f} pos={sum(1 for s in R_only if s>0)}/{len(R_only)}")
        if R_loo: print(f"  R_loo={np.mean(R_loo):+.4f} pos={sum(1 for s in R_loo if s>0)}/{len(R_loo)}")
        if R_only and R_loo:
            print(f"  R_loo / R_only = {np.mean(R_loo) / max(abs(np.mean(R_only)), 0.001):.3f}")
    
    # =====================================================================
    # SUMMARY TABLE
    # =====================================================================
    print("\n" + "=" * 80)
    print("SUMMARY TABLE: KEY METRICS AT MID-LAYER")
    print("=" * 80)
    
    print(f"\n{'Model':<12} {'Layer':<6} {'R_only':<8} {'F_only':<8} {'RF_only':<8} {'R+F':<8} {'R+F+RF':<8} {'RF_boost':<9} {'cos_RF_f':<9} {'cos_RF_r':<9} {'|cos|>0.9_f':<11} {'|cos|>0.9_r':<11}")
    print("-" * 110)
    
    for model_name in models:
        d = data[model_name]
        nl = d["n_layers"]
        mid_li = str(nl // 2)
        layer_res = d["factorial_causal"].get(mid_li, {})
        if not layer_res: continue
        
        R = [v.get("R_only_cos_shift", 0) for v in layer_res.values()]
        F = [v.get("F_only_cos_shift", 0) for v in layer_res.values()]
        RF = [v.get("RF_only_cos_shift", 0) for v in layer_res.values()]
        RpF = [v.get("R+F_cos_shift", 0) for v in layer_res.values()]
        RpFpRF = [v.get("R+F+RF_cos_shift", 0) for v in layer_res.values()]
        cos_fact = [v.get("cos_RF_factorial", 0) for v in layer_res.values()]
        cos_resid = [v.get("cos_RF_residual", 0) for v in layer_res.values()]
        
        rf_boost = np.mean(RpFpRF) - np.mean(RpF)
        extreme_f = sum(1 for v in cos_fact if abs(v) > 0.9)
        extreme_r = sum(1 for v in cos_resid if abs(v) > 0.9)
        
        print(f"{model_name:<12} L{mid_li:<5} {np.mean(R):+.4f}  {np.mean(F):+.4f}  {np.mean(RF):+.4f}  "
              f"{np.mean(RpF):+.4f}  {np.mean(RpFpRF):+.4f}  {rf_boost:+.4f}   "
              f"{np.mean(cos_fact):+.4f}   {np.mean(cos_resid):+.4f}   "
              f"{extreme_f}/{len(cos_fact):<8} {extreme_r}/{len(cos_resid)}")
    
    print("\nDone!")

if __name__ == "__main__":
    main()
