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
    # Print top-level keys
    print(f'Keys: {list(d.keys())[:5]}')
    config = d.get('config', d.get('metadata', {}))
    print(f'Config: {json.dumps(config, indent=2)[:300]}')
    
    cat_results = d.get('category_results', {})
    for cat, cr in cat_results.items():
        pv = cr.get('path_value', {})
        gs = pv.get('group_summary', {})
        lv = pv.get('logit_vs_value', {})
        hub = cr.get('hub_hidden_state', {})
        interv = cr.get('intervention', {})
        
        print(f'\n  --- {cat} ---')
        
        # Group summary
        print(f'  Group V_c(semantic):')
        for gname, gdata in gs.items():
            sem = gdata.get('avg_V_c_semantic', 0)
            lex = gdata.get('avg_V_c_lexical', 0)
            print(f'    {gname}: V_sem={sem:.3f}, V_lex={lex:.3f}')
        
        # Logit vs Value
        mr = lv.get('match_rate', 'N/A')
        sp = lv.get('spearman_r', 'N/A')
        print(f'  Logit-vs-Value match_rate: {mr}')
        print(f'  Spearman corr: {sp}')
        
        # Top path values
        pvs = pv.get('path_values', {})
        sorted_pv = sorted(pvs.items(), key=lambda x: -x[1].get('V_c_semantic', 0))[:8]
        print(f'  Top path values:')
        for name, info in sorted_pv:
            print(f'    {name}: V_sem={info.get("V_c_semantic",0):.3f}, V_lex={info.get("V_c_lexical",0):.3f}, rank={info.get("avg_logit_rank","?")}, group={info.get("group","?")}')
        
        # Hub effects on cat_logit
        he = hub.get('hub_effects', {})
        print(f'  Hub effects on cat_logit:')
        for hname, heff in he.items():
            delta = heff.get('cat_logit_delta', 0)
            print(f'    {hname}: delta={delta:+.3f}')
        
        # Intervention
        for cond in ['clean', 'best_hub', 'boost_cat_logit', 'best_hub_plus_boost']:
            ci = interv.get(cond, {})
            if ci:
                lh = ci.get('lexical_hit_rate', 'N/A')
                sh = ci.get('semantic_hit_rate', 'N/A')
                print(f'    {cond}: lex={lh}, sem={sh}')