"""GLM4/DS7B key profile extraction"""
import json, sys
sys.stdout.reconfigure(encoding='utf-8')

for model in ['glm4', 'deepseek7b']:
    path = f'results/glm5/phase486_{model}_r1.json'
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    exp1 = data.get('exp1_cross_layer_accumulation', {})
    cats_to_show = ['fruit'] if model == 'glm4' else ['fruit', 'food']
    
    for cat in cats_to_show:
        if cat not in exp1 or 'layer_profile' not in exp1[cat]:
            continue
        d = exp1[cat]
        profile = d['layer_profile']
        best = d['best_layer']
        print(f'{model.upper()} {cat} (B_c from L{best}):')
        for l in sorted(profile.keys(), key=int):
            p = profile[l]
            marker = ' <--' if int(l) == best else ''
            rd = p['resid_proj_diff']
            ad = p['attn_proj_diff']
            md = p['mlp_proj_diff']
            print(f'  L{l:>3s}  resid={rd:+8.1f}  attn={ad:+8.1f}  mlp={md:+8.1f}{marker}')
        print()
