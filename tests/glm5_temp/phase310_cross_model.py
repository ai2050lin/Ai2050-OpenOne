"""Phase 310 cross-model analysis"""
import json
import numpy as np
from pathlib import Path

for mn in ['qwen3', 'glm4', 'deepseek7b']:
    p = Path(f'results/phase310_unified_norm_causal/{mn}_unified_norm_causal.json')
    if not p.exists():
        print(f"  {mn}: FILE NOT FOUND")
        continue
    
    with open(p) as f:
        d = json.load(f)
    
    print(f'\n=== {mn} ===')
    for li in d['sample_layers']:
        lr = d['layers'][str(li)]
        cos = lr['cos_matrix_raw']
        names = lr['cos_names']
        norms = lr['direction_norms']
        gains = lr['gains']
        
        # Key cosine pairs
        def get_cos(n1, n2):
            if n1 in names and n2 in names:
                return cos[names.index(n1)][names.index(n2)]
            return None
        
        cos_OC = get_cos('O_not', 'C_pc1')
        cos_OR = get_cos('O_not', 'R')
        cos_RC = get_cos('R', 'C_pc1')
        cos_OA = get_cos('O_not', 'A')
        
        # Norms
        R_norm = norms.get('R_raw', 0)
        O_norm = norms.get('O_not_raw', 0)
        C_raw_norm = norms.get('C_raw', 0)
        Cpc1_norm = norms.get('C_pc1', 0)
        OcleanR = norms.get('O_clean_R_raw', 0)
        OcleanRC = norms.get('O_clean_RC_raw', 0)
        OcleanRCA = norms.get('O_clean_RCA_raw', 0)
        
        # Ratios
        OcleanR_ratio = OcleanR / max(O_norm, 1e-10)
        OcleanRC_ratio = OcleanRC / max(O_norm, 1e-10)
        OcleanRCA_ratio = OcleanRCA / max(O_norm, 1e-10)
        
        # Gains
        rg = gains.get('random_unit', 1e-10)
        Og = gains.get('O_not_unit', 0) / rg if rg > 1e-10 else 0
        Ocg = gains.get('O_clean_RC_unit', 0) / rg if rg > 1e-10 else 0
        Rg = gains.get('R_unit', 0) / rg if rg > 1e-10 else 0
        Cg = gains.get('C_pc1_unit', 0) / rg if rg > 1e-10 else 0
        Ag = gains.get('A_unit', 0) / rg if rg > 1e-10 else 0
        
        # Causal (alpha=2.0)
        causal = lr.get('causal_results', {})
        def get_causal(dname):
            a2 = causal.get(dname, {}).get('alpha_2.0', {})
            return a2.get('mean_delta_not', 'N/A')
        
        dn_O = get_causal('O_not_unit')
        dn_Oc = get_causal('O_clean_RC_unit')
        dn_R = get_causal('R_unit')
        dn_C = get_causal('C_pc1_unit')
        dn_A = get_causal('A_unit')
        dn_rand = get_causal('random_unit')
        
        print(f'  L{li}:')
        print(f'    cos: O-Cpc1={cos_OC:+.3f} O-R={cos_OR:+.3f} R-Cpc1={cos_RC:+.3f} O-A={cos_OA:+.3f}')
        print(f'    norms: R={R_norm:.1f} O={O_norm:.1f} C_raw={C_raw_norm:.6f} Cpc1={Cpc1_norm:.4f}')
        print(f'    ratios: OcleanR={OcleanR_ratio:.3f} OcleanRC={OcleanRC_ratio:.3f} OcleanRCA={OcleanRCA_ratio:.3f}')
        print(f'    gain: O={Og:.2f}x Oc_RC={Ocg:.2f}x R={Rg:.2f}x C={Cg:.2f}x A={Ag:.2f}x')
        print(f'    causal(a=2): O={dn_O} Oc={dn_Oc} R={dn_R} C={dn_C} A={dn_A} rand={dn_rand}')

# Print O-C conflict resolution summary
print('\n' + '='*60)
print('O-C CONFLICT RESOLUTION SUMMARY')
print('='*60)
print()
print('Phase 308: cos(O_not, C_raw) < 0.17  (C_raw norm ~0.0000003)')
print('Phase 309: cos(O_not, C_pc1) varies   (C_pc1 norm = 1.0)')
print('Phase 310: Now we know WHY:')
print()
print('C_raw is the mean of centered frame differences.')
print('Its norm is ~0 because +deltas cancel -deltas by construction.')
print('So cos(O, C_raw) is unreliable when C_raw norm is tiny.')
print()
print('C_pc1 is the first principal component of construction variation.')
print('It captures the dominant direction of frame variation.')
print('This is the meaningful construction direction.')
print()
print('Phase 308 cos(O, C_raw) ~0 was a NULL result, not evidence of orthogonality.')
print('Phase 310 cos(O, C_pc1) gives the real answer.')

# Print O_clean ratio comparison
print('\n' + '='*60)
print('O_clean RATIO COMPARISON (O_clean_R / O_raw)')
print('='*60)
for mn in ['qwen3', 'glm4', 'deepseek7b']:
    p = Path(f'results/phase310_unified_norm_causal/{mn}_unified_norm_causal.json')
    with open(p) as f:
        d = json.load(f)
    print(f'\n{mn}:')
    for li in d['sample_layers']:
        lr = d['layers'][str(li)]
        norms = lr['direction_norms']
        OcleanR_ratio = norms.get('O_clean_R_raw', 0) / max(norms.get('O_not_raw', 1e-10), 1e-10)
        OcleanRC_ratio = norms.get('O_clean_RC_raw', 0) / max(norms.get('O_not_raw', 1e-10), 1e-10)
        OcleanRCA_ratio = norms.get('O_clean_RCA_raw', 0) / max(norms.get('O_not_raw', 1e-10), 1e-10)
        print(f'  L{li}: OcleanR={OcleanR_ratio:.3f} OcleanRC={OcleanRC_ratio:.3f} OcleanRCA={OcleanRCA_ratio:.3f}')

# Print norm comparison
print('\n' + '='*60)
print('NORM COMPARISON (R / O ratio)')
print('='*60)
for mn in ['qwen3', 'glm4', 'deepseek7b']:
    p = Path(f'results/phase310_unified_norm_causal/{mn}_unified_norm_causal.json')
    with open(p) as f:
        d = json.load(f)
    mid = d['sample_layers'][len(d['sample_layers'])//2]
    lr = d['layers'][str(mid)]
    norms = lr['direction_norms']
    R_O = norms.get('R_raw', 0) / max(norms.get('O_not_raw', 1e-10), 1e-10)
    print(f'{mn} L{mid}: R/O ratio = {R_O:.2f}  R_norm={norms.get("R_raw",0):.1f}  O_norm={norms.get("O_not_raw",0):.1f}')
