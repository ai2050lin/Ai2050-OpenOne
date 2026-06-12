"""Phase 467 R1 vs R2 Consistency Check"""
import json, sys
sys.stdout.reconfigure(encoding='utf-8')

for model in ['qwen3', 'glm4', 'deepseek7b']:
    print(f'\n===== {model} =====')
    r1_path = f'results/glm5/phase467_{model}_r1.json'
    r2_path = f'results/glm5/phase467_{model}_r2.json'
    
    with open(r1_path, encoding='utf-8') as f:
        d1 = json.load(f)
    with open(r2_path, encoding='utf-8') as f:
        d2 = json.load(f)
    
    # Exp1: PC1 attribution consistency
    e1_r1 = d1.get('exp1_pc1_attribution', {})
    e1_r2 = d2.get('exp1_pc1_attribution', {})
    
    print('  Exp1 PC1 Attribution:')
    for lk in sorted(set(e1_r1.keys()) & set(e1_r2.keys())):
        lr1 = e1_r1[lk]
        lr2 = e1_r2[lk]
        if isinstance(lr1, dict) and 'pc_eigenvalues' in lr1:
            r1_pc1 = lr1['pc_eigenvalues'].get('pc1_ratio', 0)
            r2_pc1 = lr2['pc_eigenvalues'].get('pc1_ratio', 0)
            r1_cs = lr1.get('pc1_category_spread', 0)
            r2_cs = lr2.get('pc1_category_spread', 0)
            r1_nc = lr1.get('pc1_vs_norm_correlation', 0)
            r2_nc = lr2.get('pc1_vs_norm_correlation', 0)
            r1_ec = lr1.get('pc1_vs_entropy_correlation', 0)
            r2_ec = lr2.get('pc1_vs_entropy_correlation', 0)
            print(f'    {lk}: pc1_ratio R1={r1_pc1:.4f} R2={r2_pc1:.4f} delta={abs(r1_pc1-r2_pc1):.4f}, '
                  f'ent_corr R1={r1_ec:.4f} R2={r2_ec:.4f}')
    
    # Exp4: Combined directions consistency
    e4_r1 = d1.get('exp4_combined_directions', {})
    e4_r2 = d2.get('exp4_combined_directions', {})
    
    print('  Exp4 Combined Directions (selectivity):')
    for lk in sorted(set(e4_r1.keys()) & set(e4_r2.keys())):
        lr1 = e4_r1[lk]
        lr2 = e4_r2[lk]
        if isinstance(lr1, dict):
            for cat in ['vehicle', 'furniture', 'animal']:
                if cat in lr1 and cat in lr2:
                    for m in ['raw', 'no_pc1', 'disentangle']:
                        if m in lr1[cat] and m in lr2[cat]:
                            s1 = lr1[cat][m].get('selectivity', 0)
                            s2 = lr2[cat][m].get('selectivity', 0)
                            print(f'    {lk}/{cat}/{m}: R1={s1:.4f} R2={s2:.4f} delta={abs(s1-s2):.4f}')
    
    # Exp5: DS7B baseline generation check
    e5_r1 = d1.get('exp5_generation_systematic', {})
    e5_r2 = d2.get('exp5_generation_systematic', {})
    
    if model == 'deepseek7b':
        print('  DS7B Baseline Generation:')
        for cat in ['fruit', 'animal', 'vehicle', 'furniture']:
            if cat in e5_r2:
                base = e5_r2[cat].get('base_gen', '')[:70]
                print(f'    {cat}: {base}')
