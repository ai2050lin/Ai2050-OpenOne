"""
Phase 306 Cross-Model Analysis: Normalized PCA + R/C/P/N Decomposition
======================================================================
"""
import json, numpy as np
from pathlib import Path
from collections import defaultdict

RESULT_DIR = Path("results/phase306_norm_position")
MODELS = ["qwen3", "glm4", "deepseek7b"]

def load_results():
    data = {}
    for m in MODELS:
        path = RESULT_DIR / f"{m}_norm_position.json"
        if path.exists():
            data[m] = json.load(open(path, "r", encoding="utf-8"))
    return data

def main():
    data = load_results()
    print("=" * 70)
    print("Phase 306 Cross-Model Analysis: Normalized PCA + R/C/P/N Decomposition")
    print("=" * 70)
    
    # =====================================================================
    # 1. RAW vs UNIT PC1 comparison — the core finding
    # =====================================================================
    print("\n" + "=" * 70)
    print("1. RAW vs UNIT PC1 Comparison (Core Finding)")
    print("=" * 70)
    
    for m in MODELS:
        if m not in data:
            continue
        print(f"\n--- {m.upper()} ---")
        # Use middle layer for comparison
        norm_pca = data[m].get("norm_pca_results", {})
        # Find the layer closest to the model's canonical test layer
        target_layers = {"qwen3": "18", "glm4": "20", "deepseek7b": "14"}
        tl = target_layers.get(m)
        
        if tl in norm_pca:
            for role in ["adj", "verb", "noun"]:
                if role in norm_pca[tl]:
                    d = norm_pca[tl][role]
                    raw_pc1 = d["raw_pc1_var"] * 100
                    unit_pc1 = d["unit_pc1_var"] * 100
                    norm_cv = d["dev_norm_cv"]
                    corr_norm = d["corr_pc1_proj_norm"]
                    cos_mean = d["cos_pc1_mean_direction"]
                    drop_pct = (raw_pc1 - unit_pc1) / max(raw_pc1, 0.1) * 100
                    print(f"  {role}: Raw={raw_pc1:.1f}% → Unit={unit_pc1:.1f}% "
                          f"(drop={drop_pct:.0f}%), NormCV={norm_cv:.3f}, "
                          f"corr(|PC1|,norm)={corr_norm:+.3f}")
    
    # =====================================================================
    # 2. NORM CV and PC1-norm correlation across layers
    # =====================================================================
    print("\n" + "=" * 70)
    print("2. Norm Variation and PC1-Norm Correlation Across Layers")
    print("=" * 70)
    
    for m in MODELS:
        if m not in data:
            continue
        print(f"\n--- {m.upper()} ---")
        norm_pca = data[m].get("norm_pca_results", {})
        for li_str in sorted(norm_pca.keys(), key=lambda x: int(x)):
            layer_data = norm_pca[li_str]
            for role in ["adj", "verb", "noun"]:
                if role in layer_data:
                    d = layer_data[role]
                    print(f"  L{li_str} {role}: RawPC1={d['raw_pc1_var']*100:.1f}% "
                          f"UnitPC1={d['unit_pc1_var']*100:.1f}% "
                          f"NormCV={d['dev_norm_cv']:.3f} "
                          f"corr={d['corr_pc1_proj_norm']:+.3f}")
    
    # =====================================================================
    # 3. R/C/P/N Energy Budget Comparison
    # =====================================================================
    print("\n" + "=" * 70)
    print("3. R/C/P/N Energy Budget Comparison")
    print("=" * 70)
    
    for m in MODELS:
        if m not in data:
            continue
        print(f"\n--- {m.upper()} ---")
        decomp = data[m].get("decompose_results", {})
        for li_str in sorted(decomp.keys(), key=lambda x: int(x)):
            entries = decomp[li_str]
            if not entries:
                continue
            C_pcts = [v.get("C_energy_pct", 0) for v in entries.values()]
            P_pcts = [v.get("P_energy_pct", 0) for v in entries.values()]
            N_pcts = [v.get("N_energy_pct", 0) for v in entries.values()]
            U_pcts = [v.get("U_energy_pct", 0) for v in entries.values()]
            print(f"  L{li_str}: C={np.mean(C_pcts):.1f}% P={np.mean(P_pcts):.1f}% "
                  f"N={np.mean(N_pcts):.1f}% U={np.mean(U_pcts):.1f}%")
    
    # =====================================================================
    # 4. Per-role-pair decomposition (DS7B focus)
    # =====================================================================
    print("\n" + "=" * 70)
    print("4. Per-Role-Pair R/C/P/N (DS7B Focus)")
    print("=" * 70)
    
    for m in ["deepseek7b"]:
        if m not in data:
            continue
        print(f"\n--- {m.upper()} ---")
        decomp = data[m].get("decompose_results", {})
        # Use canonical layer
        target_layers = {"deepseek7b": "14"}
        tl = target_layers.get(m)
        if tl in decomp:
            rp_groups = defaultdict(list)
            for key, val in decomp[tl].items():
                rp = val.get("role_pair", "")
                rp_groups[rp].append(val)
            
            for rp in ["adj_verb", "adj_noun", "noun_verb"]:
                items = rp_groups.get(rp, [])
                if not items:
                    continue
                C_pct = np.mean([v.get("C_energy_pct", 0) for v in items])
                P_pct = np.mean([v.get("P_energy_pct", 0) for v in items])
                N_pct = np.mean([v.get("N_energy_pct", 0) for v in items])
                U_pct = np.mean([v.get("U_energy_pct", 0) for v in items])
                cos_gR = np.mean([v.get("cos_gap_R", 0) for v in items])
                print(f"  {rp}: C={C_pct:.1f}% P={P_pct:.1f}% N={N_pct:.1f}% "
                      f"U={U_pct:.1f}% cos(Gap,R)={cos_gR:+.3f}")
    
    # =====================================================================
    # 5. Causal test with normalized C direction
    # =====================================================================
    print("\n" + "=" * 70)
    print("5. Causal Test: R vs C_raw vs C_unit vs P vs FD")
    print("=" * 70)
    
    for m in MODELS:
        if m not in data:
            continue
        print(f"\n--- {m.upper()} ---")
        causal = data[m].get("causal_results", {})
        target_layers = {"qwen3": "18", "glm4": "20", "deepseek7b": "14"}
        tl = target_layers.get(m)
        
        if tl in causal:
            entries = causal[tl]
            R = [v.get("R_only_cos_shift", 0) for v in entries.values()]
            C_r = [v.get("C_raw_cos_shift", 0) for v in entries.values()]
            C_u = [v.get("C_unit_cos_shift", 0) for v in entries.values()]
            P = [v.get("P_only_cos_shift", 0) for v in entries.values()]
            FD = [v.get("full_delta_cos_shift", 0) for v in entries.values()]
            print(f"  L{tl}: R={np.mean(R):+.4f} C_raw={np.mean(C_r):+.4f} "
                  f"C_unit={np.mean(C_u):+.4f} P={np.mean(P):+.4f} FD={np.mean(FD):+.4f}")
            
            # Per-role-pair breakdown for DS7B
            if m == "deepseek7b":
                rp_groups = defaultdict(list)
                for key, val in entries.items():
                    rp = val.get("role_pair", "")
                    rp_groups[rp].append(val)
                for rp in ["adj_verb", "adj_noun", "noun_verb"]:
                    items = rp_groups.get(rp, [])
                    if not items:
                        continue
                    R_rp = np.mean([v.get("R_only_cos_shift", 0) for v in items])
                    C_r_rp = np.mean([v.get("C_raw_cos_shift", 0) for v in items])
                    C_u_rp = np.mean([v.get("C_unit_cos_shift", 0) for v in items])
                    FD_rp = np.mean([v.get("full_delta_cos_shift", 0) for v in items])
                    print(f"    {rp}: R={R_rp:+.4f} C_raw={C_r_rp:+.4f} "
                          f"C_unit={C_u_rp:+.4f} FD={FD_rp:+.4f}")
    
    # =====================================================================
    # 6. Key verdict: Is DS7B's 1D construction a norm artifact?
    # =====================================================================
    print("\n" + "=" * 70)
    print("6. VERDICT: Is DS7B's 1D Construction a Norm Artifact?")
    print("=" * 70)
    
    for m in MODELS:
        if m not in data:
            continue
        norm_pca = data[m].get("norm_pca_results", {})
        target_layers = {"qwen3": "18", "glm4": "20", "deepseek7b": "14"}
        tl = target_layers.get(m)
        
        if tl in norm_pca:
            print(f"\n  {m.upper()} L{tl}:")
            for role in ["adj", "verb", "noun"]:
                if role in norm_pca[tl]:
                    d = norm_pca[tl][role]
                    raw = d["raw_pc1_var"] * 100
                    unit = d["unit_pc1_var"] * 100
                    corr = d["corr_pc1_proj_norm"]
                    norm_cv = d["dev_norm_cv"]
                    
                    if raw - unit > 30 and corr > 0.9:
                        verdict = "NORM ARTIFACT (raw>>unit, high corr)"
                    elif raw - unit > 20:
                        verdict = "PARTIALLY NORM-DRIVEN"
                    else:
                        verdict = "GENUINE DIRECTION (raw≈unit)"
                    
                    print(f"    {role}: Raw={raw:.1f}% Unit={unit:.1f}% "
                          f"NormCV={norm_cv:.3f} corr={corr:+.3f} → {verdict}")
    
    # =====================================================================
    # 7. Summary statistics table
    # =====================================================================
    print("\n" + "=" * 70)
    print("7. Summary Statistics Table")
    print("=" * 70)
    print(f"\n{'Model':<12} {'Role':<6} {'RawPC1':<10} {'UnitPC1':<10} {'NormCV':<10} {'Corr':<10} {'Verdict'}")
    print("-" * 75)
    
    for m in MODELS:
        if m not in data:
            continue
        norm_pca = data[m].get("norm_pca_results", {})
        target_layers = {"qwen3": "18", "glm4": "20", "deepseek7b": "14"}
        tl = target_layers.get(m)
        
        if tl in norm_pca:
            for role in ["adj", "verb", "noun"]:
                if role in norm_pca[tl]:
                    d = norm_pca[tl][role]
                    raw = d["raw_pc1_var"] * 100
                    unit = d["unit_pc1_var"] * 100
                    corr = d["corr_pc1_proj_norm"]
                    norm_cv = d["dev_norm_cv"]
                    
                    if raw - unit > 30 and corr > 0.9:
                        v = "ARTIFACT"
                    elif raw - unit > 20:
                        v = "PARTIAL"
                    else:
                        v = "GENUINE"
                    
                    print(f"{m:<12} {role:<6} {raw:<10.1f} {unit:<10.1f} {norm_cv:<10.3f} {corr:<10.3f} {v}")

if __name__ == "__main__":
    main()
