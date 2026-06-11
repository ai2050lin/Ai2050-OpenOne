"""Phase 461 R1 vs R2对比 — 跨语言探针 + Exp5"""
import json, numpy as np

for m in ['qwen3', 'glm4', 'deepseek7b']:
    print(f'\n{"="*70}')
    print(f'{m}: Exp4 Cross-language probe (R1 vs R2)')
    print(f'{"="*70}')
    for r in [1, 2]:
        path = f'results/glm5/phase461_{m}_r{r}.json'
        with open(path) as f:
            d = json.load(f)
        exp4 = d.get('exp4_cross_lang_probe', {})
        if 'error' in exp4:
            print(f'  R{r}: ERROR')
            continue
        print(f'  R{r} ({d["model_info"]["n_layers"]} layers):')
        for lk in sorted(exp4.keys(), key=lambda x: int(x[1:])):
            dd = exp4[lk]
            en_acc = dd.get('en_probe_acc', 0)
            zh_acc = dd.get('zh_probe_acc_cross_lang', 0)
            avg_cos = dd.get('avg_cosine_en_zh', 0)
            n_en = dd.get('n_en_samples', 0)
            n_zh = dd.get('n_zh_samples', 0)
            center_cos = dd.get('category_center_cosine', {})
            avg_cc = np.mean(list(center_cos.values())) if center_cos else 0
            print(f'    {lk}: en={en_acc:.2f}, zh_cross={zh_acc:.2f}, cos={avg_cos:.3f}, '
                  f'center_cos={avg_cc:.3f} (n={n_en}/{n_zh})')

    print(f'\n  {m}: Exp5 Large beta synthesis (R2)')
    path = f'results/glm5/phase461_{m}_r2.json'
    with open(path) as f:
        d = json.load(f)
    exp5 = d.get('exp5_large_beta', {})
    if 'error' in exp5:
        print(f'  ERROR')
        continue
    for case in exp5:
        case_data = exp5[case]
        for lk in sorted(case_data.keys(), key=lambda x: int(x[1:])):
            dd = case_data[lk]
            base = dd.get('base_margin', 0)
            line = f'    {case} {lk}: base={base:.2f}'
            for beta in [10, 50]:
                sel = dd.get(f'beta{beta}_selectivity', 0)
                dt = dd.get(f'beta{beta}_delta_target', 0)
                dc = dd.get(f'beta{beta}_delta_comp', 0)
                line += f', b{beta}:sel={sel:.2f}(dt={dt:.2f},dc={dc:.2f})'
            print(line)

    print(f'\n  {m}: Exp1 Shared/Private ratio (R2, key layers)')
    exp1 = d.get('exp1_wdown_row', {})
    if 'error' not in exp1:
        for cat in ['fruit', 'tool']:
            if cat not in exp1:
                continue
            for lk in sorted(exp1[cat].keys(), key=lambda x: int(x[1:])):
                dd = exp1[cat][lk]
                if 'error' in dd:
                    continue
                ratio = dd.get('shared_private_ratio', 0)
                overlap = dd.get('overlap_top_k', 0)
                corr = dd.get('shared_private_corr', 0)
                if ratio < 10000:  # skip L0 artifacts
                    print(f'    {cat} {lk}: ratio={ratio:.1f}, overlap={overlap}/20, corr={corr:.3f}')
