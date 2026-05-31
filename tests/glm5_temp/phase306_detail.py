import json, numpy as np
d = json.load(open('results/phase306_norm_position/deepseek7b_norm_position.json','r',encoding='utf-8'))
pca = d['norm_pca_results']
print('DS7B Unit PC1 across layers:')
for li in sorted(pca.keys(), key=lambda x: int(x)):
    r = pca[li]
    for role in ['adj','verb','noun']:
        if role in r:
            raw = r[role]['raw_pc1_var']*100
            unit = r[role]['unit_pc1_var']*100
            ncv = r[role]['dev_norm_cv']
            corr = r[role]['corr_pc1_proj_norm']
            print(f'  L{li} {role}: raw={raw:.1f}% unit={unit:.1f}% NormCV={ncv:.3f} corr={corr:+.3f}')

# Also check Qwen3 and GLM4
for model in ['qwen3', 'glm4']:
    d2 = json.load(open(f'results/phase306_norm_position/{model}_norm_position.json','r',encoding='utf-8'))
    pca2 = d2['norm_pca_results']
    print(f'\n{model.upper()} Unit PC1 across layers:')
    for li in sorted(pca2.keys(), key=lambda x: int(x)):
        r = pca2[li]
        for role in ['adj','verb','noun']:
            if role in r:
                raw = r[role]['raw_pc1_var']*100
                unit = r[role]['unit_pc1_var']*100
                ncv = r[role]['dev_norm_cv']
                corr = r[role]['corr_pc1_proj_norm']
                print(f'  L{li} {role}: raw={raw:.1f}% unit={unit:.1f}% NormCV={ncv:.3f} corr={corr:+.3f}')

# Causal test: C_unit vs C_raw comparison for DS7B
print('\n\nDS7B Causal test C_unit vs C_raw across layers:')
d3 = json.load(open('results/phase306_norm_position/deepseek7b_norm_position.json','r',encoding='utf-8'))
causal = d3.get('causal_results', {})
for li in sorted(causal.keys(), key=lambda x: int(x)):
    entries = causal[li]
    R = np.mean([v.get('R_only_cos_shift',0) for v in entries.values()])
    C_r = np.mean([v.get('C_raw_cos_shift',0) for v in entries.values()])
    C_u = np.mean([v.get('C_unit_cos_shift',0) for v in entries.values()])
    P = np.mean([v.get('P_only_cos_shift',0) for v in entries.values()])
    FD = np.mean([v.get('full_delta_cos_shift',0) for v in entries.values()])
    print(f'  L{li}: R={R:+.4f} C_raw={C_r:+.4f} C_unit={C_u:+.4f} P={P:+.4f} FD={FD:+.4f}')

# Per-role-pair for DS7B
print('\nDS7B per-role-pair causal (L14):')
if '14' in causal:
    from collections import defaultdict
    rp_groups = defaultdict(list)
    for key, val in causal['14'].items():
        rp = val.get('role_pair', '')
        rp_groups[rp].append(val)
    for rp in ['adj_verb', 'adj_noun', 'noun_verb']:
        items = rp_groups.get(rp, [])
        if items:
            R = np.mean([v.get('R_only_cos_shift',0) for v in items])
            C_r = np.mean([v.get('C_raw_cos_shift',0) for v in items])
            C_u = np.mean([v.get('C_unit_cos_shift',0) for v in items])
            FD = np.mean([v.get('full_delta_cos_shift',0) for v in items])
            print(f'  {rp}: R={R:+.4f} C_raw={C_r:+.4f} C_unit={C_u:+.4f} FD={FD:+.4f}')
