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
    print(f'Top keys: {list(d.keys())}')
    
    cat_results = d.get('category_results', {})
    for cat, cr in cat_results.items():
        pv = cr.get('path_value', {})
        
        # Print raw structure first
        print(f'\n  --- {cat} ---')
        print(f'  path_value keys: {list(pv.keys())}')
        gs = pv.get('group_summary', {})
        print(f'  group_summary: {json.dumps(gs, indent=2)[:500]}')
        
        lv = pv.get('logit_vs_value', {})
        print(f'  logit_vs_value: {json.dumps(lv, indent=2)[:500]}')
        
        hub = cr.get('hub_hidden_state', {})
        print(f'  hub_hidden_state keys: {list(hub.keys())}')
        he = hub.get('hub_effects', {})
        print(f'  hub_effects: {json.dumps(he, indent=2)[:500]}')
        
        interv = cr.get('intervention', {})
        print(f'  intervention: {json.dumps(interv, indent=2)[:600]}')