import json
for model in ['qwen3', 'deepseek7b', 'glm4']:
    with open(f'results/glm5/phase464_{model}_r1.json', encoding='utf-8') as f:
        d = json.load(f)
    print(f'\n=== {model} ===')
    
    # Exp1
    if 'exp1_orthogonal_fix' in d:
        print('Exp1: 正交分解修复')
        for lk, items in d['exp1_orthogonal_fix'].items():
            if not items: continue
            first = list(items.values())[0]
            cos_v = first['cos_sem_lang']
            nr = first['sem_only_ratio_fixed']
            orr = first['sem_only_ratio_old']
            th = first['theoretical_ratio']
            er = first['ratio_error']
            print(f'  {lk}: cos={cos_v:.4f}, NEW_ratio={nr:.4f}, OLD_ratio={orr:.4f}, theory={th:.4f}, err={er:.6f}')
    
    # Exp3
    if 'exp3_cross_category_holdout' in d:
        print('Exp3: 跨类别holdout')
        for key in sorted(d['exp3_cross_category_holdout'].keys()):
            val = d['exp3_cross_category_holdout'][key]
            cat = val['category']
            beta = val['beta']
            sel = val['avg_selectivity']
            n = val['n_test_objects']
            print(f'  {key}: cat={cat} beta={beta} sel={sel:.3f} n={n}')
    
    # Exp6
    if 'exp6_strategy_indicators' in d:
        print('Exp6: 策略指标')
        for k, v in d['exp6_strategy_indicators'].items():
            if isinstance(v, float):
                print(f'  {k}: {v:.4f}')
            else:
                print(f'  {k}: {v}')
