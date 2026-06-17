import sys; sys.stdout.reconfigure(encoding='utf-8')
import json, os

results_dir = 'd:/AI2050/Ai2050-OpenOne/results/glm5_phase514_path_value'

for model in ['qwen3', 'glm4', 'deepseek7b']:
    fp = os.path.join(results_dir, f'phase514_{model}_path_value.json')
    if not os.path.exists(fp):
        print(f'{model}: NO RESULTS')
        continue
    d = json.load(open(fp, 'r', encoding='utf-8'))
    print(f'\n=== {model} ===')
    print(f'Test objects: {d["config"]["n_test_objects"]}')
    print(f'Steps: {d["config"]["steps"]}')
    
    for cat, cr in d['category_results'].items():
        pv = cr.get('path_value', {})
        gs = pv.get('group_summary', {})
        lv = pv.get('logit_vs_value', {})
        hub = cr.get('hub_hidden_state', {})
        interv = cr.get('intervention', {})
        
        print(f'\n  --- {cat} ---')
        
        # Group summary
        print(f'  Group V_c(semantic):')
        for gname, gdata in gs.items():
            print(f'    {gname}: V_sem={gdata["avg_V_c_semantic"]:.3f}, V_lex={gdata["avg_V_c_lexical"]:.3f}')
        
        # Logit vs Value
        print(f'  Logit-vs-Value match_rate: {lv.get("match_rate", "N/A")}')
        print(f'  Spearman corr: {lv.get("spearman_r", "N/A")}')
        
        # Top path values
        pvs = pv.get('path_values', {})
        sorted_pv = sorted(pvs.items(), key=lambda x: -x[1]['V_c_semantic'])[:8]
        print(f'  Top path values:')
        for name, info in sorted_pv:
            print(f'    {name}: V_sem={info["V_c_semantic"]:.3f}, V_lex={info["V_c_lexical"]:.3f}, rank={info["avg_logit_rank"]:.0f}, group={info["group"]}')
        
        # Hub effects on cat_logit
        he = hub.get('hub_effects', {})
        print(f'  Hub effects on cat_logit:')
        for hname, heff in he.items():
            delta = heff['cat_logit_delta']
            print(f'    {hname}: delta={delta:+.3f}')
        
        # Intervention
        for cond in ['clean', 'best_hub', 'boost_cat_logit', 'best_hub_plus_boost']:
            ci = interv.get(cond, {})
            if ci:
                print(f'    {cond}: lex={ci.get("lexical_hit_rate","N/A")}, sem={ci.get("semantic_hit_rate","N/A")}')