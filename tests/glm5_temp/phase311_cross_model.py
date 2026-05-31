"""Phase 311 cross-model analysis"""
import json
import numpy as np
from pathlib import Path

print("="*70)
print("PHASE 311: NORM-MATCHED CAUSAL TEST - CROSS-MODEL ANALYSIS")
print("="*70)

for mn in ['qwen3', 'glm4', 'deepseek7b']:
    p = Path(f'results/phase311_norm_matched_causal/{mn}_norm_matched_causal.json')
    if not p.exists():
        print(f"\n{mn}: FILE NOT FOUND")
        continue
    
    with open(p) as f:
        d = json.load(f)
    
    print(f"\n{'='*70}")
    print(f"  {mn}")
    print(f"{'='*70}")
    
    for li in d['test_layers']:
        lr = d['layers'][str(li)]
        dn = lr['direction_norms']
        cd = lr['causal_data'].get('scale_0.1', {})
        
        print(f"\n  Layer {li}:")
        print(f"  Direction norms:")
        for k in ['R', 'C_pc1', 'O_not', 'O_clean_R', 'O_clean_RC', 'O_clean_RCA', 'A']:
            if k in dn:
                ratio = dn[k] / max(dn.get('O_not', 1e-10), 1e-10)
                print(f"    {k:>15}: {dn[k]:>10.2f}  (ratio_to_O={ratio:.3f})")
        
        print(f"\n  Causal effects (scale=0.1):")
        print(f"  {'Direction':>15} {'PatchNorm':>10} {'Δnot':>8} {'Δsad':>8} {'Δhappy':>8} {'Δnot/norm':>10}")
        for dname in ['O_not', 'O_clean_R', 'O_clean_RC', 'O_clean_RCA', 'R', 'C_pc1', 'A', 'random']:
            if dname in cd:
                eff = cd[dname]['effects']
                pn = cd[dname]['patch_norm']
                dn_val = eff.get('not', 0)
                ds = eff.get(' sad', 0)
                dh = eff.get(' happy', 0)
                # Causal efficiency: effect per unit norm
                eff_per_norm = dn_val / max(pn, 1e-10)
                print(f"    {dname:>15} {pn:>10.1f} {dn_val:>+8.3f} {ds:>+8.3f} {dh:>+8.3f} {eff_per_norm:>+10.5f}")

# Key comparison: normalized causal efficiency
print(f"\n{'='*70}")
print(f"KEY COMPARISON: Normalized Causal Efficiency (Δnot / patch_norm)")
print(f"{'='*70}")

for mn in ['qwen3', 'glm4', 'deepseek7b']:
    p = Path(f'results/phase311_norm_matched_causal/{mn}_norm_matched_causal.json')
    with open(p) as f:
        d = json.load(f)
    
    print(f"\n{mn}:")
    for li in d['test_layers']:
        lr = d['layers'][str(li)]
        cd = lr['causal_data'].get('scale_0.1', {})
        print(f"  L{li}:")
        for dname in ['O_not', 'O_clean_R', 'O_clean_RC', 'R', 'A', 'random']:
            if dname in cd:
                eff = cd[dname]['effects']
                pn = cd[dname]['patch_norm']
                dn_val = eff.get('not', 0)
                eff_per_norm = dn_val / max(pn, 1e-10)
                print(f"    {dname:>15}: Δnot/norm = {eff_per_norm:>+.6f}")

# DS7B specific: compare O_not vs O_clean_R at same norm
print(f"\n{'='*70}")
print(f"DS7B: O_not vs O_clean_R at same injection norm")
print(f"{'='*70}")

p = Path('results/phase311_norm_matched_causal/deepseek7b_norm_matched_causal.json')
with open(p) as f:
    d = json.load(f)

for li in d['test_layers']:
    lr = d['layers'][str(li)]
    cd = lr['causal_data']
    
    print(f"\n  L{li}:")
    # Compare at different scales
    for sf_key in cd:
        if 'O_not' in cd[sf_key] and 'O_clean_R' in cd[sf_key]:
            on = cd[sf_key]['O_not']
            ocr = cd[sf_key]['O_clean_R']
            rand = cd[sf_key].get('random', {})
            
            on_eff = on['effects'].get('not', 0) / max(on['patch_norm'], 1e-10)
            ocr_eff = ocr['effects'].get('not', 0) / max(ocr['patch_norm'], 1e-10)
            rand_eff = rand.get('effects', {}).get('not', 0) / max(rand.get('patch_norm', 1e-10), 1e-10)
            
            print(f"    {sf_key}: O_not_eff={on_eff:+.5f} O_clean_R_eff={ocr_eff:+.5f} random_eff={rand_eff:+.5f}")
            print(f"            O_clean_R/O_not ratio = {ocr_eff/max(abs(on_eff),1e-10):.1f}x")
