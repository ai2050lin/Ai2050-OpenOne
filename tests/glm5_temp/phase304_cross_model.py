"""
Phase 304 Cross-Model Analysis: Construction + Gap Decomposition
================================================================
Compare construction identification and gap decomposition across models.
"""
import sys, json, os
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
from collections import defaultdict

RESULT_DIR = "results/phase304_construction_gap"
MODELS = ["qwen3", "glm4", "deepseek7b"]

def load_results():
    data = {}
    for m in MODELS:
        path = os.path.join(RESULT_DIR, f"{m}_construction_gap.json")
        if os.path.exists(path):
            data[m] = json.load(open(path, 'r', encoding='utf-8'))
    return data

def main():
    data = load_results()
    print("=" * 80)
    print("Phase 304 Cross-Model Analysis: Construction + Gap Decomposition")
    print("=" * 80)
    
    # =====================================================================
    # 1. WITHIN-ROLE FRAME PCA: Explained Variance
    # =====================================================================
    print("\n" + "=" * 60)
    print("1. WITHIN-ROLE FRAME PCA: Explained Variance by Role")
    print("=" * 60)
    
    for m in MODELS:
        if m not in data:
            continue
        nl = data[m]["n_layers"]
        mid_li = str(nl // 2)
        pca = data[m].get("pca_results", {}).get(mid_li, {})
        
        print(f"\n  {m.upper()} L{mid_li}:")
        for role in ["adj", "verb", "noun"]:
            if role in pca:
                vr = pca[role].get("explained_var_ratio", [])
                n_obs = pca[role].get("n_observations", 0)
                n_tok = pca[role].get("n_tokens", 0)
                top5 = [f"{v:.3f}" for v in vr[:5]] if vr else ["N/A"]
                print(f"    {role}: n_obs={n_obs}, n_tok={n_tok}, "
                      f"top-5 var_ratio={top5}")
                print(f"      PC1 explains {vr[0]*100:.1f}% of frame variation within role")
    
    # =====================================================================
    # 2. CROSS-ROLE SUBSPACE ANGLES
    # =====================================================================
    print("\n" + "=" * 60)
    print("2. CROSS-ROLE SUBSPACE ANGLES (Frame Subspace Sharing)")
    print("=" * 60)
    
    for m in MODELS:
        if m not in data:
            continue
        nl = data[m]["n_layers"]
        mid_li = str(nl // 2)
        angles = data[m].get("subspace_angles", {}).get(mid_li, {})
        
        print(f"\n  {m.upper()} L{mid_li}:")
        for pair, adata in angles.items():
            min_a = adata.get("min_angle_deg", 0)
            mean_a = adata.get("mean_angle_deg", 0)
            all_a = adata.get("angles_deg", [])
            print(f"    {pair}: min_angle={min_a:.1f}° mean_angle={mean_a:.1f}° "
                  f"angles={[f'{a:.1f}°' for a in all_a[:5]]}")
    
    # =====================================================================
    # 3. CAUSAL EFFECT COMPARISON
    # =====================================================================
    print("\n" + "=" * 60)
    print("3. CAUSAL EFFECT COMPARISON (Mid Layer)")
    print("=" * 60)
    
    metrics = ["R_only_cos_shift", "full_delta_cos_shift", "Gap_only_cos_shift",
               "C_only_cos_shift", "R+C_cos_shift", "avg_random_shift"]
    
    for m in MODELS:
        if m not in data:
            continue
        nl = data[m]["n_layers"]
        mid_li = str(nl // 2)
        cr = data[m].get("causal_results", {}).get(mid_li, {})
        
        if not cr:
            continue
        
        print(f"\n  {m.upper()} L{mid_li} ({len(cr)} tokens):")
        for metric in metrics:
            vals = [v.get(metric) for v in cr.values() if v.get(metric) is not None]
            if vals:
                pos = sum(1 for v in vals if v > 0)
                print(f"    {metric:25s}: {np.mean(vals):+.4f} ± {np.std(vals):.4f} "
                      f"pos={pos}/{len(vals)} ({pos/len(vals)*100:.0f}%)")
        
        # Gap decomposition
        gap_norms = [v.get("gap_norm", 0) for v in cr.values()]
        C_norms = [v.get("C_vec_norm", 0) for v in cr.values()]
        U_norms = [v.get("U_vec_norm", 0) for v in cr.values()]
        cos_gR = [v.get("cos_gap_R", 0) for v in cr.values()]
        cos_gC = [v.get("cos_gap_C", 0) for v in cr.values()]
        C_energy = [v.get("C_proj_energy", 0) for v in cr.values()]
        U_pct = [v.get("U_norm_pct", 0) for v in cr.values()]
        sign_agree = [v.get("R_fd_same_sign", 0) for v in cr.values()]
        
        print(f"\n    Gap decomposition:")
        print(f"      gap_norm:    {np.mean(gap_norms):.4f}")
        print(f"      C_norm:      {np.mean(C_norms):.4f}")
        print(f"      U_norm:      {np.mean(U_norms):.4f} ({np.mean(U_pct):.1f}% of gap)")
        print(f"      cos(Gap,R):  {np.mean(cos_gR):+.4f}  ← {'ANTI-R!' if np.mean(cos_gR) < -0.3 else 'weak'}")
        print(f"      cos(Gap,C):  {np.mean(cos_gC):+.4f}  ← {'PRO-C!' if np.mean(cos_gC) > 0.5 else 'weak'}")
        print(f"      C_proj_energy: {np.mean(C_energy):.4f}")
        print(f"      R-FD sign agreement: {sum(sign_agree)}/{len(sign_agree)} ({sum(sign_agree)/len(sign_agree)*100:.0f}%)")
        
        # R+C vs R_only boost
        R_only = [v.get("R_only_cos_shift", 0) for v in cr.values() if v.get("R_only_cos_shift") is not None]
        RpC = [v.get("R+C_cos_shift", 0) for v in cr.values() if v.get("R+C_cos_shift") is not None]
        if R_only and RpC:
            boost = np.mean(RpC) - np.mean(R_only)
            print(f"      R+C boost over R_only: {boost:+.4f} ({boost/max(abs(np.mean(R_only)),0.01)*100:+.1f}%)")
    
    # =====================================================================
    # 4. PER-ROLE-PAIR BREAKDOWN
    # =====================================================================
    print("\n" + "=" * 60)
    print("4. PER-ROLE-PAIR BREAKDOWN")
    print("=" * 60)
    
    for m in MODELS:
        if m not in data:
            continue
        nl = data[m]["n_layers"]
        mid_li = str(nl // 2)
        cr = data[m].get("causal_results", {}).get(mid_li, {})
        
        if not cr:
            continue
        
        print(f"\n  {m.upper()} L{mid_li}:")
        
        rp_groups = defaultdict(list)
        for v in cr.values():
            rp_groups[v.get("role_pair", "")].append(v)
        
        for rp in ["adj_verb", "adj_noun", "noun_verb"]:
            items = rp_groups.get(rp, [])
            if not items:
                continue
            
            R_only = [v.get("R_only_cos_shift", 0) for v in items if v.get("R_only_cos_shift") is not None]
            FD = [v.get("full_delta_cos_shift", 0) for v in items if v.get("full_delta_cos_shift") is not None]
            RpC = [v.get("R+C_cos_shift", 0) for v in items if v.get("R+C_cos_shift") is not None]
            cos_gR = [v.get("cos_gap_R", 0) for v in items if v.get("cos_gap_R") is not None]
            cos_gC = [v.get("cos_gap_C", 0) for v in items if v.get("cos_gap_C") is not None]
            sign_agree = [v.get("R_fd_same_sign", 0) for v in items]
            
            print(f"\n    {rp} ({len(items)} tokens):")
            if R_only:
                print(f"      R_only:   {np.mean(R_only):+.4f}")
            if FD:
                print(f"      full_delta: {np.mean(FD):+.4f}")
            if RpC:
                print(f"      R+C:      {np.mean(RpC):+.4f}")
            if cos_gR:
                print(f"      cos(Gap,R): {np.mean(cos_gR):+.4f}")
            if cos_gC:
                print(f"      cos(Gap,C): {np.mean(cos_gC):+.4f}")
            if sign_agree:
                print(f"      R-FD sign agree: {sum(sign_agree)}/{len(sign_agree)} ({sum(sign_agree)/len(sign_agree)*100:.0f}%)")
    
    # =====================================================================
    # 5. BOOTSTRAP 95% CI COMPARISON
    # =====================================================================
    print("\n" + "=" * 60)
    print("5. BOOTSTRAP 95% CI COMPARISON")
    print("=" * 60)
    
    for m in MODELS:
        if m not in data:
            continue
        boot = data[m].get("bootstrap", {})
        
        print(f"\n  {m.upper()}:")
        for key in sorted(boot.keys()):
            if "::" in key and "all::" in key:
                rp, metric = key.split("::", 1)
                br = boot[key]
                ci_excl_0 = (br["ci_low"] > 0 or br["ci_high"] < 0)
                star = "***" if ci_excl_0 else ""
                print(f"    {metric:25s}: {br['mean']:+.4f} CI=[{br['ci_low']:+.4f}, {br['ci_high']:+.4f}] {star}")
    
    # =====================================================================
    # 6. DS7B DEEP DIVE: Anti-R Gap Analysis
    # =====================================================================
    print("\n" + "=" * 60)
    print("6. DS7B DEEP DIVE: Anti-R Gap Analysis")
    print("=" * 60)
    
    m = "deepseek7b"
    if m in data:
        nl = data[m]["n_layers"]
        mid_li = str(nl // 2)
        cr = data[m].get("causal_results", {}).get(mid_li, {})
        
        if cr:
            # Categorize tokens by cos(Gap, R)
            anti_R = []  # cos_gR < -0.5
            neutral = []  # -0.5 <= cos_gR <= 0.5
            pro_R = []  # cos_gR > 0.5
            
            for v in cr.values():
                cgR = v.get("cos_gap_R", 0)
                if cgR < -0.5:
                    anti_R.append(v)
                elif cgR > 0.5:
                    pro_R.append(v)
                else:
                    neutral.append(v)
            
            print(f"\n  DS7B L{mid_li} cos(Gap,R) distribution:")
            print(f"    Anti-R (< -0.5):  {len(anti_R)} tokens")
            print(f"    Neutral (-0.5~0.5): {len(neutral)} tokens")
            print(f"    Pro-R (> 0.5):    {len(pro_R)} tokens")
            
            # Anti-R tokens by role pair
            rp_antiR = defaultdict(list)
            for v in anti_R:
                rp_antiR[v.get("role_pair", "")].append(v)
            
            print(f"\n    Anti-R tokens by role pair:")
            for rp, items in sorted(rp_antiR.items()):
                R_shifts = [v.get("R_only_cos_shift", 0) for v in items]
                FD_shifts = [v.get("full_delta_cos_shift", 0) for v in items]
                print(f"      {rp}: {len(items)} tokens, "
                      f"R_shift={np.mean(R_shifts):+.3f}, FD_shift={np.mean(FD_shifts):+.3f}")
            
            # Key finding: what is the Gap aligned with?
            print(f"\n    Gap alignment (anti-R tokens only):")
            cos_gC = [v.get("cos_gap_C", 0) for v in anti_R if v.get("cos_gap_C") is not None]
            C_energy = [v.get("C_proj_energy", 0) for v in anti_R if v.get("C_proj_energy") is not None]
            if cos_gC:
                print(f"      cos(Gap, C): {np.mean(cos_gC):+.4f}")
            if C_energy:
                print(f"      C_proj_energy: {np.mean(C_energy):.4f}")
            
            # How much of Gap is unresolved?
            U_pct = [v.get("U_norm_pct", 0) for v in anti_R if v.get("U_norm_pct") is not None]
            if U_pct:
                print(f"      U(unresolved): {np.mean(U_pct):.1f}%")
            
            # R_only vs R+C vs full_delta for anti-R tokens
            R_only = [v.get("R_only_cos_shift", 0) for v in anti_R if v.get("R_only_cos_shift") is not None]
            RpC = [v.get("R+C_cos_shift", 0) for v in anti_R if v.get("R+C_cos_shift") is not None]
            FD = [v.get("full_delta_cos_shift", 0) for v in anti_R if v.get("full_delta_cos_shift") is not None]
            if R_only:
                print(f"      R_only causal:   {np.mean(R_only):+.4f}")
            if RpC:
                print(f"      R+C causal:      {np.mean(RpC):+.4f}")
            if FD:
                print(f"      full_delta causal: {np.mean(FD):+.4f}")
    
    # =====================================================================
    # 7. KEY CROSS-MODEL COMPARISON TABLE
    # =====================================================================
    print("\n" + "=" * 60)
    print("7. KEY CROSS-MODEL COMPARISON TABLE")
    print("=" * 60)
    
    header = f"{'Metric':25s} {'Qwen3':>10s} {'GLM4':>10s} {'DS7B':>10s}"
    print(f"\n  {header}")
    print(f"  {'-'*60}")
    
    metrics_compare = [
        ("R_only_cos_shift", "R_only causal"),
        ("full_delta_cos_shift", "full_delta causal"),
        ("C_only_cos_shift", "C(construction) causal"),
        ("R+C_cos_shift", "R+C causal"),
        ("avg_random_shift", "Random baseline"),
    ]
    
    for metric_key, metric_label in metrics_compare:
        row = f"  {metric_label:25s}"
        for m in MODELS:
            if m not in data:
                row += f" {'N/A':>10s}"
                continue
            nl = data[m]["n_layers"]
            mid_li = str(nl // 2)
            cr = data[m].get("causal_results", {}).get(mid_li, {})
            vals = [v.get(metric_key) for v in cr.values() if v.get(metric_key) is not None]
            if vals:
                row += f" {np.mean(vals):+10.4f}"
            else:
                row += f" {'N/A':>10s}"
        print(row)
    
    # Gap decomposition comparison
    print(f"\n  {'Gap Decomposition':25s}")
    gap_metrics = [
        ("gap_norm", "Gap norm"),
        ("cos_gap_R", "cos(Gap, R)"),
        ("cos_gap_C", "cos(Gap, C)"),
        ("C_proj_energy", "C projection energy"),
        ("U_norm_pct", "U(unresolved) %"),
    ]
    
    for metric_key, metric_label in gap_metrics:
        row = f"  {metric_label:25s}"
        for m in MODELS:
            if m not in data:
                row += f" {'N/A':>10s}"
                continue
            nl = data[m]["n_layers"]
            mid_li = str(nl // 2)
            cr = data[m].get("causal_results", {}).get(mid_li, {})
            vals = [v.get(metric_key) for v in cr.values() if v.get(metric_key) is not None]
            if vals:
                row += f" {np.mean(vals):+10.4f}"
            else:
                row += f" {'N/A':>10s}"
        print(row)
    
    # R-FD sign agreement
    row = f"  {'R-FD sign agree %':25s}"
    for m in MODELS:
        if m not in data:
            row += f" {'N/A':>10s}"
            continue
        nl = data[m]["n_layers"]
        mid_li = str(nl // 2)
        cr = data[m].get("causal_results", {}).get(mid_li, {})
        vals = [v.get("R_fd_same_sign", 0) for v in cr.values()]
        if vals:
            pct = sum(vals) / len(vals) * 100
            row += f" {pct:9.0f}%"
        else:
            row += f" {'N/A':>10s}"
    print(row)
    
    print("\n" + "=" * 80)
    print("Phase 304 Cross-Model Analysis Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
