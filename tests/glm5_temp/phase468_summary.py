import json, sys
sys.stdout.reconfigure(encoding='utf-8')

for model in ['qwen3', 'glm4', 'deepseek7b']:
    with open(f'results/glm5/phase468_{model}_r1.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f'\n===== {model.upper()} =====')
    
    # Exp1 Summary
    print('\n--- Exp1: PC1 causal verification ---')
    for layer_key, lr in data['exp1_pc1_causal'].items():
        s = lr.get('summary', {})
        pc1_de = s.get('pc1_mean_delta_entropy', '?')
        rand_de = s.get('random_mean_delta_entropy', '?')
        ratio = s.get('pc1_vs_random_entropy_ratio', '?')
        is_ent = s.get('is_entropy_axis', '?')
        print(f'  {layer_key}: PC1_dEnt={pc1_de}, rand_dEnt={rand_de}, ratio={ratio}, is_entropy={is_ent}')
    
    # Exp2 Summary
    print('\n--- Exp2: PC1 decomposition ---')
    for layer_key, lr in data['exp2_pc1_decomposition'].items():
        cr = lr.get('component_ratios', {})
        ec = lr.get('pc1_entropy_correlation', '?')
        pc = lr.get('pc1_position_correlation', '?')
        ra = lr.get('readout_alignment', '?')
        print(f'  {layer_key}: ent_corr={ec}, pos_corr={pc}, readout={ra}')
        print(f'    comps: ent={cr.get("entropy","?")}, pos={cr.get("position","?")}, '
              f'tmpl={cr.get("template","?")}, rdout={cr.get("readout","?")}, res={cr.get("residual","?")}')
    
    # Exp3 Summary
    print('\n--- Exp3: Purification strategy search ---')
    for layer_key, lr in data['exp3_purification_search'].items():
        print(f'  {layer_key}:')
        for cat, cr in lr.items():
            best = cr.get('best_strategy', '?')
            sel = cr.get('best_selectivity', '?')
            sig = cr.get('significant', '?')
            print(f'    {cat}: best={best}({sel}), sig={sig}')
    
    # Exp4 Summary
    print('\n--- Exp4: Template robustness ---')
    for tmpl_key, tr in data['exp4_template_robustness'].items():
        mr = tr.get('math_trigger_rate', '?')
        print(f'  {tmpl_key}: math_trigger_rate={mr}')
