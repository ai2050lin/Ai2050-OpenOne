"""
Phase 301 Cross-Model Analysis: Orthogonal R/F Decomposition
=============================================================
Compare raw vs orthogonal R/F effects across Qwen3/GLM4/DS7B
"""
import sys, json, numpy as np
from pathlib import Path
from collections import defaultdict

RESULT_DIR = Path("results/phase301_orthogonal_rf")

def load_results(model):
    path = RESULT_DIR / f"{model}_orthogonal_rf.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def analyze():
    models = ["qwen3", "glm4", "deepseek7b"]
    all_data = {m: load_results(m) for m in models}
    
    print("=" * 80)
    print("PHASE 301: Cross-Model Orthogonal R/F Decomposition Analysis")
    print("=" * 80)
    
    # ===== 1. Core comparison per layer =====
    print("\n" + "=" * 80)
    print("1. CORE COMPARISON: Raw vs Orthogonal Bundle per Layer")
    print("=" * 80)
    
    for model in models:
        data = all_data[model]
        nl = data["n_layers"]
        print(f"\n--- {model} (n_layers={nl}) ---")
        print(f"  {'Layer':>6} | {'R_raw':>8} | {'R_clean':>8} | {'R+F_raw':>8} | {'R+F_clean':>8} | {'delta':>8} | {'Interact':>8} | {'OrthoNorm':>9} | {'Random':>7} | {'cos(R,F)':>8}")
        print(f"  {'-'*6}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*9}-+-{'-'*7}-+-{'-'*8}")
        
        for li_str in sorted(data["orthogonal_rf_causal"].keys(), key=int):
            layer_res = data["orthogonal_rf_causal"][li_str]
            if not layer_res: continue
            
            metrics = defaultdict(list)
            for key, v in layer_res.items():
                for m in ["R_raw_cos_shift", "R_clean_cos_shift", "R_raw+F_raw_cos_shift", 
                          "R_clean+F_clean_cos_shift", "interaction_cos_shift", 
                          "ortho_dir+norm_cos_shift", "avg_random_shift", "cos_RF"]:
                    val = v.get(m)
                    if val is not None:
                        metrics[m].append(val)
            
            R_raw = np.mean(metrics["R_raw_cos_shift"]) if metrics["R_raw_cos_shift"] else 0
            R_clean = np.mean(metrics["R_clean_cos_shift"]) if metrics["R_clean_cos_shift"] else 0
            RF_raw = np.mean(metrics["R_raw+F_raw_cos_shift"]) if metrics["R_raw+F_raw_cos_shift"] else 0
            RF_clean = np.mean(metrics["R_clean+F_clean_cos_shift"]) if metrics["R_clean+F_clean_cos_shift"] else 0
            interact = np.mean(metrics["interaction_cos_shift"]) if metrics["interaction_cos_shift"] else 0
            ortho_n = np.mean(metrics["ortho_dir+norm_cos_shift"]) if metrics["ortho_dir+norm_cos_shift"] else 0
            rand = np.mean(metrics["avg_random_shift"]) if metrics["avg_random_shift"] else 0
            cos_rf = np.mean(metrics["cos_RF"]) if metrics["cos_RF"] else 0
            delta = RF_clean - RF_raw
            
            marker = ""
            if RF_raw < 0 and RF_clean > 0:
                marker = " *** FIX ***"
            elif RF_raw > 0 and RF_clean < RF_raw:
                marker = " (worse)"
            elif RF_raw > 0 and RF_clean > RF_raw:
                marker = " (better)"
            
            print(f"  L{li_str:>5} | {R_raw:>+8.4f} | {R_clean:>+8.4f} | {RF_raw:>+8.4f} | {RF_clean:>+8.4f} | {delta:>+8.4f} | {interact:>+8.4f} | {ortho_n:>+9.4f} | {rand:>+7.4f} | {cos_rf:>+8.4f}{marker}")
    
    # ===== 2. Per-token detail at mid layer =====
    print("\n" + "=" * 80)
    print("2. PER-TOKEN DETAIL AT MID LAYER (Key Test)")
    print("=" * 80)
    
    mid_layers = {"qwen3": "18", "glm4": "20", "deepseek7b": "14"}
    
    for model in models:
        data = all_data[model]
        li = mid_layers[model]
        layer_res = data["orthogonal_rf_causal"].get(li, {})
        if not layer_res: continue
        
        print(f"\n--- {model} Layer {li} ---")
        print(f"  {'Token':>10} | {'cos(R,F)':>8} | {'R_raw':>7} | {'R_clean':>8} | {'R+F_raw':>8} | {'R+F_clean':>10} | {'Interact':>8} | {'OrthNorm':>8}")
        print(f"  {'-'*10}-+-{'-'*8}-+-{'-'*7}-+-{'-'*8}-+-{'-'*8}-+-{'-'*10}-+-{'-'*8}-+-{'-'*8}")
        
        for key, v in sorted(layer_res.items()):
            token = v["token"]
            cos_rf = v.get("cos_RF", 0)
            r_raw = v.get("R_raw_cos_shift", 0)
            r_clean = v.get("R_clean_cos_shift", 0)
            rf_raw = v.get("R_raw+F_raw_cos_shift", 0)
            rf_clean = v.get("R_clean+F_clean_cos_shift", 0)
            interact = v.get("interaction_cos_shift", 0)
            ortho_n = v.get("ortho_dir+norm_cos_shift", 0)
            
            marker = ""
            if abs(cos_rf) > 0.9:
                marker = " <<<"
            elif abs(cos_rf) > 0.7:
                marker = " <<"
            
            print(f"  {key:>10} | {cos_rf:>+8.3f} | {r_raw:>+7.3f} | {r_clean:>+8.3f} | {rf_raw:>+8.3f} | {rf_clean:>+10.3f} | {interact:>+8.3f} | {ortho_n:>+8.3f}{marker}")
    
    # ===== 3. cos(R,F) distribution =====
    print("\n" + "=" * 80)
    print("3. cos(R,F) DISTRIBUTION — R/F Overlap Comparison")
    print("=" * 80)
    
    for model in models:
        data = all_data[model]
        cos_vals = []
        for li_str, layer_res in data["orthogonal_rf_causal"].items():
            for key, v in layer_res.items():
                cv = v.get("cos_RF")
                if cv is not None:
                    cos_vals.append((int(li_str), v["token"], v.get("role_pair", ""), cv))
        
        if not cos_vals: continue
        cvals = [c[3] for c in cos_vals]
        print(f"\n--- {model} ---")
        print(f"  Mean cos(R,F) = {np.mean(cvals):+.4f}")
        print(f"  Std           = {np.std(cvals):.4f}")
        print(f"  Range         = [{min(cvals):+.4f}, {max(cvals):+.4f}]")
        print(f"  |cos|>0.9     = {sum(1 for c in cvals if abs(c)>0.9)}/{len(cvals)} ({100*sum(1 for c in cvals if abs(c)>0.9)/len(cvals):.0f}%)")
        print(f"  |cos|>0.7     = {sum(1 for c in cvals if abs(c)>0.7)}/{len(cvals)} ({100*sum(1 for c in cvals if abs(c)>0.7)/len(cvals):.0f}%)")
        
        # Per role_pair breakdown
        rp_cos = defaultdict(list)
        for li, tok, rp, cv in cos_vals:
            rp_cos[rp].append(cv)
        print(f"  By role_pair:")
        for rp in ["adj_verb", "adj_noun", "noun_verb"]:
            if rp in rp_cos:
                rpc = rp_cos[rp]
                print(f"    {rp}: mean={np.mean(rpc):+.4f}, |cos|>0.9={sum(1 for c in rpc if abs(c)>0.9)}/{len(rpc)}")
    
    # ===== 4. Orthogonal norm test =====
    print("\n" + "=" * 80)
    print("4. ORTHOGONAL NORM TEST — Is Norm an Independent Channel?")
    print("=" * 80)
    print("  ortho_dir+norm = direction orthogonal to R,F + correct norm")
    print("  If this is significantly > random baseline, norm is an independent channel.")
    print()
    
    for model in models:
        data = all_data[model]
        print(f"  {model}:")
        for li_str in sorted(data["orthogonal_rf_causal"].keys(), key=int):
            layer_res = data["orthogonal_rf_causal"][li_str]
            ortho_shifts = [v.get("ortho_dir+norm_cos_shift") for v in layer_res.values()
                           if v.get("ortho_dir+norm_cos_shift") is not None]
            rand_shifts = [v.get("avg_random_shift", 0) for v in layer_res.values()]
            
            if ortho_shifts and rand_shifts:
                avg_ortho = np.mean(ortho_shifts)
                avg_rand = np.mean(rand_shifts)
                ratio = avg_ortho / max(abs(avg_rand), 1e-6)
                is_norm_channel = "YES" if avg_ortho > avg_rand * 1.5 and avg_ortho > 0.05 else "NO"
                print(f"    L{li_str}: ortho_norm={avg_ortho:+.4f} random={avg_rand:+.4f} ratio={ratio:.2f}x norm_channel={is_norm_channel}")
    
    # ===== 5. Deep layer analysis =====
    print("\n" + "=" * 80)
    print("5. DEEP LAYER ANALYSIS — R/F Effects at Deep Layers")
    print("=" * 80)
    
    for model in models:
        data = all_data[model]
        nl = data["n_layers"]
        deep_li = str(nl - 2)
        layer_res = data["orthogonal_rf_causal"].get(deep_li, {})
        if not layer_res: continue
        
        print(f"\n--- {model} Layer {deep_li} (deep) ---")
        for key, v in sorted(layer_res.items()):
            cos_rf = v.get("cos_RF", 0)
            r_raw = v.get("R_raw_cos_shift", 0)
            rf_raw = v.get("R_raw+F_raw_cos_shift", 0)
            rf_clean = v.get("R_clean+F_clean_cos_shift", 0)
            print(f"  {key}: cos(R,F)={cos_rf:+.3f} R_raw={r_raw:+.3f} bundle_raw={rf_raw:+.3f} bundle_clean={rf_clean:+.3f}")
    
    # ===== 6. KEY DIAGNOSTIC =====
    print("\n" + "=" * 80)
    print("6. KEY DIAGNOSTIC: Does Orthogonalization Fix DS7B's Negative Bundle?")
    print("=" * 80)
    
    for model in models:
        data = all_data[model]
        nl = data["n_layers"]
        mid_li = str(nl // 2)
        layer_res = data["orthogonal_rf_causal"].get(mid_li, {})
        if not layer_res: continue
        
        raw_bundles = [v.get("R_raw+F_raw_cos_shift") for v in layer_res.values()
                      if v.get("R_raw+F_raw_cos_shift") is not None]
        clean_bundles = [v.get("R_clean+F_clean_cos_shift") for v in layer_res.values()
                        if v.get("R_clean+F_clean_cos_shift") is not None]
        interactions = [v.get("interaction_cos_shift") for v in layer_res.values()
                       if v.get("interaction_cos_shift") is not None]
        
        avg_raw = np.mean(raw_bundles) if raw_bundles else 0
        avg_clean = np.mean(clean_bundles) if clean_bundles else 0
        avg_interact = np.mean(interactions) if interactions else 0
        
        if avg_raw < 0 and avg_clean > 0:
            verdict = "ORTHOGONALIZATION FIXES NEGATIVE BUNDLE"
        elif avg_raw < 0 and avg_clean < 0:
            verdict = "ORTHOGONALIZATION DOES NOT FIX — token-specific coding"
        elif avg_raw > 0 and avg_clean > avg_raw:
            verdict = "Orthogonalization improves positive bundle"
        elif avg_raw > 0 and avg_clean < avg_raw:
            verdict = "Orthogonalization weakens positive bundle — R/F overlap is beneficial"
        else:
            verdict = "No significant change"
        
        print(f"\n  {model} (Layer {mid_li}):")
        print(f"    R_raw+F_raw     = {avg_raw:+.4f}")
        print(f"    R_clean+F_clean = {avg_clean:+.4f}")
        print(f"    Interaction     = {avg_interact:+.4f}")
        print(f"    VERDICT: {verdict}")

if __name__ == "__main__":
    analyze()
