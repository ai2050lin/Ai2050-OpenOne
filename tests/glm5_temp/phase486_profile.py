"""Phase 486 Exp1 详细跨层累积剖面"""
import json, sys
sys.stdout.reconfigure(encoding='utf-8')

for model in ['qwen3', 'glm4', 'deepseek7b']:
    path = f'results/glm5/phase486_{model}_r1.json'
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    exp1 = data.get('exp1_cross_layer_accumulation', {})
    if 'error' in exp1:
        continue
    
    print(f'\n=== {model.upper()} Exp1: Full Layer Profile ===')
    for cat in exp1:
        if not isinstance(exp1[cat], dict) or 'layer_profile' not in exp1[cat]:
            continue
        d = exp1[cat]
        profile = d['layer_profile']
        best = d['best_layer']
        print(f'\n  {cat} (B_c from L{best}):')
        header = f"  {'Layer':>6s} {'resid_diff':>12s} {'attn_diff':>12s} {'mlp_diff':>12s} {'resid_cat':>12s} {'mlp_cat':>12s}"
        print(header)
        for l in sorted(profile.keys(), key=int):
            p = profile[l]
            marker = ' <-- best' if int(l) == best else ''
            print(f"  L{l:>4s} {p['resid_proj_diff']:+12.2f} {p['attn_proj_diff']:+12.2f} {p['mlp_proj_diff']:+12.2f} {p['resid_proj_cat']:+12.2f} {p['mlp_proj_cat']:+12.2f}{marker}")
