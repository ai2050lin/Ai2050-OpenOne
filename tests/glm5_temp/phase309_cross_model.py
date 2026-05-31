"""Phase 309 cross-model analysis"""
import json, numpy as np
from pathlib import Path
from collections import defaultdict

models = ["qwen3", "glm4", "deepseek7b"]
model_layers = {"qwen3": 18, "glm4": 20, "deepseek7b": 14}

all_data = {}
for m in models:
    p = Path(f"results/phase309_subspace_map/{m}_subspace_map.json")
    if p.exists():
        all_data[m] = json.load(open(p, encoding='utf-8'))

# ============ KEY LAYER OVERLAP ============
print("=" * 80)
print("SUBSPACE OVERLAP: O_proj_R, O_proj_C, R_proj_C, O_proj_A")
print("=" * 80)

for m in models:
    if m not in all_data:
        continue
    d = all_data[m]
    overlap = d.get("overlap_results", {})
    ml = str(model_layers[m])
    
    print(f"\n{m.upper()} (key layer L{ml}):")
    if ml in overlap:
        data = overlap[ml]
        keys = ["O_proj_R", "O_proj_C", "O_proj_C_pc1", "O_proj_A", "O_proj_S",
                "R_proj_C", "R_proj_O_not", "R_proj_A"]
        for k in keys:
            if k in data:
                print(f"  {k:20s}: {data[k]:.3f}")

# ============ CROSS-LAYER TRENDS ============
print("\n" + "=" * 80)
print("CROSS-LAYER OVERLAP TRENDS")
print("=" * 80)

for m in models:
    if m not in all_data:
        continue
    d = all_data[m]
    overlap = d.get("overlap_results", {})
    print(f"\n{m.upper()}:")
    print(f"  {'Layer':>6s} {'O→R':>8s} {'O→C':>8s} {'O→Cpc1':>8s} {'O→A':>8s} {'R→C':>8s} {'R→O':>8s}")
    for li_str in sorted(overlap.keys(), key=int):
        data = overlap[li_str]
        row = f"  L{li_str:>4s}"
        for k in ["O_proj_R", "O_proj_C", "O_proj_C_pc1", "O_proj_A", "R_proj_C", "R_proj_O_not"]:
            v = data.get(k, 0)
            row += f" {v:>8.3f}"
        print(row)

# ============ COSINE MATRIX ============
print("\n" + "=" * 80)
print("COSINE MATRIX AT KEY LAYER")
print("=" * 80)

for m in models:
    if m not in all_data:
        continue
    d = all_data[m]
    cos = d.get("cos_matrices", {})
    ml = str(model_layers[m])
    
    # Find nearest layer
    if ml not in cos:
        available = [int(k) for k in cos.keys()]
        ml = str(min(available, key=lambda x: abs(x - model_layers[m])))
    
    if ml in cos:
        names = cos[ml]["names"]
        matrix = np.array(cos[ml]["matrix"])
        
        print(f"\n{m.upper()} (L{ml}):")
        # Select key functions
        key_funcs = ["R", "C", "C_pc1", "O_not", "O_maybe", "O_must", "O_can", "O_never", 
                      "A_antonym", "S_narrow", "S_wide", "S_scope"]
        key_idx = [i for i, n in enumerate(names) if n in key_funcs]
        key_names = [names[i] for i in key_idx]
        
        header = "".join([f"{n:>10s}" for n in key_names])
        print(f"{'':>10s}{header}")
        for ii, i in enumerate(key_idx):
            row = f"{key_names[ii]:>10s}"
            for jj, j in enumerate(key_idx):
                row += f"{matrix[i,j]:>+10.3f}"
            print(row)

# ============ INDEPENDENCE CAUSAL EFFECTS ============
print("\n" + "=" * 80)
print("INDEPENDENCE CAUSAL EFFECTS: clean vs raw directions")
print("=" * 80)

for m in models:
    if m not in all_data:
        continue
    d = all_data[m]
    ind = d.get("independence_results", {})
    ml = str(model_layers[m])
    
    if ml not in ind:
        available = [int(k) for k in ind.keys()]
        ml = str(min(available, key=lambda x: abs(x - model_layers[m])))
    
    if ml in ind:
        data = ind[ml]
        print(f"\n{m.upper()} (L{ml}):")
        print(f"  {'Direction':>20s} {'→not':>10s} {'→happy':>10s} {'→sad':>10s}")
        for dname in ["R_raw", "O_raw", "C_raw", "A_raw",
                      "R_clean_OC", "O_clean_RC", "C_clean_RO", "O_clean_RCA",
                      "random"]:
            if dname in data and data[dname]:
                eff = data[dname]
                not_eff = eff.get("not", 0)
                happy_eff = eff.get("happy", 0)
                sad_eff = eff.get("sad", 0)
                print(f"  {dname:>20s} {not_eff:>+10.4f} {happy_eff:>+10.4f} {sad_eff:>+10.4f}")
