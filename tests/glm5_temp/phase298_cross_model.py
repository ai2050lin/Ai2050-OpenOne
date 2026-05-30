"""
Phase 298 Cross-Model Comparison
=================================
Compare role subspace structure and causal effectiveness across Qwen3/GLM4/DS7B.
"""
import json, numpy as np
from pathlib import Path
from collections import defaultdict

RESULT_DIR = Path("results/phase298_role_subspace")
MODELS = ["qwen3", "glm4", "deepseek7b"]

def load_results():
    data = {}
    for m in MODELS:
        p = RESULT_DIR / f"{m}_role_subspace.json"
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                data[m] = json.load(f)
    return data

def compare_subspace_dimensionality(data):
    print("=" * 80)
    print("COMPARISON 1: Role Subspace Dimensionality")
    print("=" * 80)
    print(f"{'Model':<12} {'n_layers':>8} {'Layer':>6} {'top1%':>6} {'top3%':>6} {'top5%':>6} {'dim50':>6} {'dim80':>6} {'dim95':>6} {'mean_norm':>10}")
    
    for m in MODELS:
        if m not in data:
            continue
        d = data[m]
        nl = d["n_layers"]
        rpca = d["role_increment_pca"]
        sample_layers = sorted(set([0, nl//4, nl//2, 3*nl//4, nl-1, nl]) & set(int(k) for k in rpca.keys()))
        
        for li in sample_layers:
            li_str = str(li)
            if li_str in rpca:
                r = rpca[li_str]
                print(f"{m:<12} {nl:>8} {li:>6} {r['top1_var']*100:>6.1f} {r['top3_var']*100:>6.1f} "
                      f"{r['top5_var']*100:>6.1f} {r['dim_50']:>6} {r['dim_80']:>6} {r['dim_95']:>6} "
                      f"{r['mean_increment_norm']:>10.1f}")

def compare_frame_dimensionality(data):
    print("\n" + "=" * 80)
    print("COMPARISON 2: Frame Subspace Dimensionality")
    print("=" * 80)
    print(f"{'Model':<12} {'Layer':>6} {'top1%':>6} {'top3%':>6} {'dim50':>6} {'dim80':>6} {'dim95':>6}")
    
    for m in MODELS:
        if m not in data:
            continue
        d = data[m]
        nl = d["n_layers"]
        fpca = d["frame_increment_pca"]
        sample_layers = sorted(set([0, nl//4, nl//2, 3*nl//4, nl-1, nl]) & set(int(k) for k in fpca.keys()))
        
        for li in sample_layers:
            li_str = str(li)
            if li_str in fpca:
                r = fpca[li_str]
                print(f"{m:<12} {li:>6} {r['top1_var']*100:>6.1f} {r['top3_var']*100:>6.1f} "
                      f"{r['dim_50']:>6} {r['dim_80']:>6} {r['dim_95']:>6}")

def compare_overlap(data):
    print("\n" + "=" * 80)
    print("COMPARISON 3: Role-Frame Subspace Overlap")
    print("=" * 80)
    print(f"{'Model':<12} {'Layer':>6} {'avg_angle°':>10} {'min_angle°':>10} {'top1_cos':>10}")
    
    for m in MODELS:
        if m not in data:
            continue
        d = data[m]
        nl = d["n_layers"]
        ovlp = d["subspace_overlap"]
        sample_layers = sorted(set([0, nl//4, nl//2, 3*nl//4, nl-1, nl]) & set(int(k) for k in ovlp.keys()))
        
        for li in sample_layers:
            li_str = str(li)
            if li_str in ovlp:
                r = ovlp[li_str]
                print(f"{m:<12} {li:>6} {r['avg_principal_angle_deg']:>10.1f} "
                      f"{r['min_principal_angle_deg']:>10.1f} {r['top1_cos_overlap']:>+10.4f}")

def compare_loo(data):
    print("\n" + "=" * 80)
    print("COMPARISON 4: Cross-Token Generalization (Leave-One-Out)")
    print("=" * 80)
    print(f"{'Model':<12} {'Layer':>6} {'avg_LOO_cos':>12} {'std_LOO_cos':>12} {'n_tokens':>10}")
    
    for m in MODELS:
        if m not in data:
            continue
        d = data[m]
        nl = d["n_layers"]
        loo = d["cross_token_generalization"]
        sample_layers = sorted(set([0, nl//4, nl//2, 3*nl//4, nl-1, nl]) & set(int(k) for k in loo.keys()))
        
        for li in sample_layers:
            li_str = str(li)
            if li_str in loo:
                r = loo[li_str]
                print(f"{m:<12} {li:>6} {r['avg_loo_cosine']:>+12.4f} {r['std_loo_cosine']:>12.4f} {r['n_tokens']:>10}")

def compare_causal(data):
    print("\n" + "=" * 80)
    print("COMPARISON 5: Causal Direction Test (mean_delta direction)")
    print("=" * 80)
    
    # Aggregate by model: average across all token pairs and mid-layers
    for m in MODELS:
        if m not in data:
            continue
        d = data[m]
        causal = d["causal_direction_test"]
        nl = d["n_layers"]
        mid = nl // 2
        
        # Collect mid-layer results (within ±3 of mid-layer)
        cos_shifts = []
        random_shifts = []
        specificities = []
        kl_shifts = []
        
        for key, layer_results in causal.items():
            for layer_key, r in layer_results.items():
                if "mean_delta" not in layer_key:
                    continue
                li_str = layer_key.split("_")[0][1:]
                try:
                    li = int(li_str)
                except:
                    continue
                if abs(li - mid) <= 5 and li > 0:
                    cos_shifts.append(r["cos_shift_toward_target"])
                    random_shifts.append(r["random_cos_shift"])
                    specificities.append(r["specificity_ratio"])
                    kl_shifts.append(r["kl_shift_toward_target"])
        
        if cos_shifts:
            n_positive = sum(1 for s in cos_shifts if s > 0)
            print(f"\n  {m} (mid-layer ±5, n={len(cos_shifts)}):")
            print(f"    avg cos_shift: {np.mean(cos_shifts):+.6f} (positive = toward target)")
            print(f"    avg random_shift: {np.mean(random_shifts):+.6f}")
            print(f"    avg specificity: {np.mean(specificities):.1f}x")
            print(f"    positive shift rate: {n_positive}/{len(cos_shifts)} = {n_positive/len(cos_shifts)*100:.0f}%")
            print(f"    avg KL_shift: {np.mean(kl_shifts):+.6f}")
            
            # Per token-pair breakdown
            print(f"\n    Per token-pair:")
            for key in sorted(causal.keys()):
                pair_shifts = []
                for layer_key, r in causal[key].items():
                    if "mean_delta" in layer_key:
                        li_str = layer_key.split("_")[0][1:]
                        try:
                            li = int(li_str)
                        except:
                            continue
                        if abs(li - mid) <= 5 and li > 0:
                            pair_shifts.append(r["cos_shift_toward_target"])
                if pair_shifts:
                    avg_shift = np.mean(pair_shifts)
                    status = "+" if avg_shift > 0 else "-"
                    print(f"      {key:20s}: avg_shift={avg_shift:+.6f} {status}")

def key_findings(data):
    print("\n" + "=" * 80)
    print("KEY FINDINGS SUMMARY")
    print("=" * 80)
    
    for m in MODELS:
        if m not in data:
            continue
        d = data[m]
        nl = d["n_layers"]
        rpca = d["role_increment_pca"]
        fpca = d["frame_increment_pca"]
        loo = d["cross_token_generalization"]
        
        # Mid-layer stats
        mid = nl // 2
        mid_str = str(mid)
        
        r_top1 = rpca.get(mid_str, {}).get("top1_var", 0)
        r_dim50 = rpca.get(mid_str, {}).get("dim_50", 0)
        r_dim80 = rpca.get(mid_str, {}).get("dim_80", 0)
        f_top1 = fpca.get(mid_str, {}).get("top1_var", 0)
        f_dim50 = fpca.get(mid_str, {}).get("dim_50", 0)
        loo_cos = loo.get(mid_str, {}).get("avg_loo_cosine", 0)
        
        print(f"\n  {m} (mid L{mid}):")
        print(f"    Role subspace: top1={r_top1*100:.1f}%, dim50={r_dim50}, dim80={r_dim80}")
        print(f"    Frame subspace: top1={f_top1*100:.1f}%, dim50={f_dim50}")
        print(f"    Cross-token LOO cos: {loo_cos:+.4f}")
        
        # Causal summary
        causal = d["causal_direction_test"]
        cos_shifts = []
        for key, layer_results in causal.items():
            for layer_key, r in layer_results.items():
                if "mean_delta" in layer_key:
                    li_str = layer_key.split("_")[0][1:]
                    try:
                        li = int(li_str)
                    except:
                        continue
                    if abs(li - mid) <= 5 and li > 0:
                        cos_shifts.append(r["cos_shift_toward_target"])
        if cos_shifts:
            n_pos = sum(1 for s in cos_shifts if s > 0)
            print(f"    Causal shift: avg={np.mean(cos_shifts):+.6f}, positive_rate={n_pos}/{len(cos_shifts)}={n_pos/len(cos_shifts)*100:.0f}%")


if __name__ == "__main__":
    data = load_results()
    compare_subspace_dimensionality(data)
    compare_frame_dimensionality(data)
    compare_overlap(data)
    compare_loo(data)
    compare_causal(data)
    key_findings(data)
