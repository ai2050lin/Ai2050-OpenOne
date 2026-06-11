"""Phase 461 R1 详细分析 — Exp4跨语言+Exp2跨类别"""
import json

for m in ['qwen3', 'glm4', 'deepseek7b']:
    path = f'results/glm5/phase461_{m}_r1.json'
    with open(path) as f:
        d = json.load(f)
    
    print(f'\n{"="*60}')
    print(f'{m} Exp4: Cross-language probe')
    print(f'{"="*60}')
    exp4 = d.get('exp4_cross_lang_probe', {})
    if 'error' in exp4:
        print(f'  ERROR: {exp4["error"]}')
        continue
    for lk in sorted(exp4.keys(), key=lambda x: int(x[1:])):
        dd = exp4[lk]
        en_acc = dd.get('en_probe_acc', 0)
        zh_acc = dd.get('zh_probe_acc_cross_lang', 0)
        avg_cos = dd.get('avg_cosine_en_zh', 0)
        print(f'  {lk}: en_acc={en_acc:.2f}, zh_cross_acc={zh_acc:.2f} (random=0.25), avg_cos={avg_cos:.3f}')
        if 'category_center_cosine' in dd:
            for c, v in dd['category_center_cosine'].items():
                print(f'    {c}_center_cos={v:.3f}')

    print(f'\n{m} Exp3: Translate encoding (apple)')
    print(f'{"-"*60}')
    exp3 = d.get('exp3_translate', {})
    if 'apple' in exp3:
        apple = exp3['apple']
        for lk in sorted(apple.keys(), key=lambda x: int(x[1:])):
            dd = apple[lk]
            en2zh = dd.get('en2zh_diff_norm', 0)
            zh2en = dd.get('zh2en_diff_norm', 0)
            cross = dd.get('en2zh_vs_zh2en_diff_cos', 0)
            en_norm = dd.get('en_norm', 0)
            trans_norm = dd.get('trans_en2zh_norm', 0)
            zh_norm = dd.get('zh_norm', 0)
            print(f'  {lk}: en2zh_diff={en2zh:.1f}, zh2en_diff={zh2en:.1f}, cross_cos={cross:.3f}')
            print(f'       en_norm={en_norm:.1f}, trans_en2zh_norm={trans_norm:.1f}, zh_norm={zh_norm:.1f}')

    print(f'\n{m} Exp2: Effective rank & variance')
    print(f'{"-"*60}')
    exp2 = d.get('exp2_cross_object_diff', {})
    for cat in ['fruit', 'animal']:
        if cat not in exp2:
            continue
        for lk in sorted(exp2[cat].keys(), key=lambda x: int(x[1:])):
            dd = exp2[cat][lk]
            eff = dd.get('effective_rank_90pct', 0)
            ve = dd.get('variance_explained', [])
            avg_pc = dd.get('avg_private_cosine_offdiag', 0)
            ve_str = ', '.join([f'{v:.3f}' for v in ve[:4]]) if ve else 'N/A'
            print(f'  {cat} {lk}: eff_rank={eff}, avg_priv_cos={avg_pc:.3f}, var_expl=[{ve_str}]')
            # PC1 projections
            pc1 = dd.get('pc1_projections', {})
            if pc1:
                proj_str = ', '.join([f'{k}={v:.2f}' for k, v in sorted(pc1.items())])
                print(f'         PC1: {proj_str}')
